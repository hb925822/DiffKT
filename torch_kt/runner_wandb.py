import multiprocessing as mp
import os
import shutil
import traceback

import pandas as pd
import wandb
from torch.utils.data import DataLoader

from torch_kt.runner import run_preprocess, get_model, KtTrainner
from torch_kt.utils import root_dir
from torch_kt.wandb_utils import download_wandb_dataset


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB__EXECUTABLE"
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class KtTrainnerWandb(KtTrainner):
    def __init__(self, wandb_run, **kwargs):
        super().__init__(**kwargs)
        self.wandb_run = wandb_run

    def on_epoch_end(self, epoch, logs=None):
        self.wandb_run.log(logs)


def train(dataset_dir, dataset_dir_d, log_dir, job_type, group, fast_run, name=None, **kwargs):
    test_result = {}
    try:
        reset_wandb_env()
        with wandb.init(job_type=job_type, group=group, name=name, config=kwargs) as wandb_run:
            wb_conf = wandb_run.config
            keyargs = dict(wb_conf)
            model = get_model(wb_conf.model_name, wb_conf.data_name, keyargs)
            feature_names, label_names = model.inputs_specs
            d = dataset_dir  # DirDataset(dataset_dir)
            trdata_loader, vadata_loader, tedata_loader = d.split_generator(dataset_dir_d, feature_names,
                                                                            label_names,
                                                                            max_len=wb_conf.max_len)
            train_dataloader = DataLoader(trdata_loader, batch_size=wb_conf.batch_size, shuffle=True)
            test_dataloader = DataLoader(vadata_loader, batch_size=wb_conf.batch_size)
            val_dataloader = DataLoader(tedata_loader, batch_size=wb_conf.batch_size)
            model.compile_model(optimizer=wb_conf.optimizer, lr=wb_conf.lr)
            trainner = KtTrainnerWandb(wandb_run, log_dir=log_dir,
                                       max_epochs=wb_conf.max_epochs,
                                       device=wb_conf.device, patience=wb_conf.patience,
                                       valid_interval=wb_conf.valid_interval)
            test_logs = trainner.fit(model, train_dataloader, test_dataloader, val_dataloader)
            wandb_run.summary.update(test_logs)
            test_result = test_logs
            wandb_run.finish()
    except Exception as e:
        traceback.print_exc()
        test_result = None
    finally:
        return test_result


def _postprocess_for_train(all_results, log_dir, wandb_run):
    all_df = pd.DataFrame(all_results)
    best_idx = all_df["auc"].idxmax()
    mean_result = all_df.mean()
    mean_result["p"] = all_df["auc"].std()
    mean_result.to_csv(os.path.join(log_dir, "test_result.csv"), index=False, encoding='utf-8',
                       float_format='%.5f')
    final_result = mean_result.to_dict()
    ##############config 有的不需要单独保存
    # final_result.update(
    #     {'dataname': wb_conf.data_name, 'item_type': wb_conf.item_type,
    #      'model': wb_conf.model_name})
    wandb_run.summary.update(final_result)
    # my_table = wandb.Table(dataframe=pd.DataFrame([final_result]))
    # wandb_run.log({"EvalResult": my_table})
    # wandb_run.log_code(include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
    # ma = make_model_atifact(os.path.join(log_dir, f'k{best_idx}'), wb_conf.model_name)
    # wandb_run.log_artifact(ma, aliases=[f'{wb_conf.data_name}', "best"])


def wandb_main(model_name, data_name, data_base=None, logs_base=None, max_len=-1, folds=5, fast_run=False, **kwargs):
    if data_base is None:
        data_base = os.path.join(root_dir(), "data")
    data_artifact, _ = download_wandb_dataset(data_base, data_name)
    folds_params, log_dir, data_set, run_config, model_config = run_preprocess(model_name, data_name, data_base,
                                                                               logs_base, max_len, folds, **kwargs)
    args = run_config
    try:
        if folds > 0:
            settings = wandb.Settings(job_source="artifact") if os.name == "nt" else None
            with wandb.init(job_type="test" if fast_run else "cross-val", group=data_name, config=args,
                            settings=settings) as wandb_run:
                wb_conf = wandb_run.config
                wandb_run.use_artifact(data_artifact)
                all_results = []
                job_type = "test" if fast_run else "cross-val-sub"
                group =  data_name + "_single"
                with mp.Pool(processes=5) as pools:
                    for i,(dataset_d, log_d) in enumerate(folds_params):
                        result = pools.apply(train, (
                            data_set, dataset_d, log_d, job_type, group, fast_run,f'{wandb_run.name}f{i}'), kwds=wb_conf.as_dict())
                        if result:
                            all_results.append(result)
                        print('process end')
                if len(all_results) == folds:
                    _postprocess_for_train(all_results, log_dir, wandb_run)
                    wandb_run.finish()
                else:
                    wandb_run.finish(-1)
                # shutil.rmtree(log_dir, ignore_errors=True)  # wandb运行的不需要本地保存
                print("over")
                return 0 if len(all_results) == folds else -1
        else:
            dataset_d, log_d = folds_params[0]
            job_type = "test" if fast_run else "model-val"
            group = data_name
            train(data_set, dataset_d, log_d, job_type, group, fast_run, **args)
    except RuntimeError:
        traceback.print_exc()
        shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    wandb_main("DKT", "assist0910", "../data", "../logs", folds=5, max_len=100, batch_size=32, max_epochs=5, lr=0.01,
               optimizer="adam",
               patience=5, valid_interval=1, device="cuda")
