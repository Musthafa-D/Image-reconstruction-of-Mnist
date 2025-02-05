from learner import Learner, Conditional_Learner
from ccbdl.parameter_optimizer.optuna_base import BaseOptunaParamOptimizer
from ccbdl.utils import DEVICE
from datetime import datetime, timedelta
from data_loader import prepare_data
from Networks.encoder_decoder import Encoder_Decoder, Conditional_Encoder_Decoder
import optuna
import os
import ccbdl


class Optuna(BaseOptunaParamOptimizer):
    def __init__(self,
                 study_config: dict,
                 optimize_config: dict,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 config,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        # get sampler and pruner for parent class
        if "sampler" in study_config.keys():
            if hasattr(optuna.samplers, study_config["sampler"]["name"]):
                sampler = getattr(
                    optuna.samplers, study_config["sampler"]["name"])()
        else:
            sampler = optuna.samplers.TPESampler()

        if "pruner" in study_config.keys():
            if hasattr(optuna.pruners, study_config["pruner"]["name"]):
                pruner = getattr(
                    optuna.pruners, study_config["pruner"]["name"])()
        else:
            pruner = None

        super().__init__(study_config["direction"], study_config["study_name"], study_path,
                         study_config["number_of_trials"], data_config["task"], comment, 
                         study_config["optimization_target"],
                         sampler, pruner, config_path, debug, logging)

        self.optimize_config = optimize_config
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.study_config = study_config
        
        self.result_folder = study_path
        self.config = config

        optuna.logging.disable_default_handler()
        self.create_study()

        self.create_study()
        self.study_name = study_config["study_name"]
        self.durations = []

    def _objective(self, trial):
        start_time = datetime.now()

        print("\n\n******* Trial " + str(trial.number) +
              " has started" + "*******\n")
        trial_folder = f'trial_{trial.number}'
        trial_path = os.path.join(self.result_folder, trial_folder)
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        # suggest parameters
        suggested = self._suggest_parameters(self.optimize_config, trial)

        self.learner_config["learning_rate_exp"] = suggested["learning_rate_exp"]
        self.learner_config["optimizer"] = suggested["optimizer"]

        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)
        
        if self.learner_config["en_de_model"] == "en_de":
            model = Encoder_Decoder(**self.network_config["en_de"]).to(DEVICE)               
            self.learner = Learner(trial_path,
                                   model,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
            
        elif self.learner_config["en_de_model"] == "conditional_en_de":
            model = Conditional_Encoder_Decoder(**self.network_config["conditional_en_de"]).to(DEVICE)              
            self.learner = Conditional_Learner(trial_path,
                                   model,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
        
        else:
            raise ValueError(
                f"Invalid value for model: {self.learner_config['model']}, it should be ddpm, cddpm, latent_dm or conditional_latent_dm only.")

        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write("Current config:-\n")
        self.learner.parameter_storage.store(self.config)

        self.learner.parameter_storage.write(
            f"Start Time of encoder decoder training and evaluation in this Trial {trial.number}: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.store(suggested, header="suggested_parameters")
        self.learner.parameter_storage.write("\n")

        print(f"\n\n******* Trial {trial.number} is completed*******")

        end_time = datetime.now()

        self.learner.parameter_storage.write(
            f"End Time of encoder decoder training and evaluation in this Trial {trial.number}: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations.append(self.duration_trial)

        self.learner.parameter_storage.write(
            f"Duration of encoder decoder training and evaluation in this Trial {trial.number}: {str(self.duration_trial)[:-7]}\n")

        return self.learner.best_values[self.optimization_target]

    def start_study(self):
        self.study.optimize(self._objective, n_trials=self.number_of_trials,)

    def eval_study(self):
        if self.logging:
            self.logger.info("evaluating study")   
        start_time = datetime.now()
        
        parameter_storage = ccbdl.storages.storages.ParameterStorage(
            self.result_folder, file_name="study_info.txt")

        parameter_storage.write("******* Summary " +
                                "of " + self.study_name + " *******")
        pruned_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if self.logging:
            self.logger.info("creating optuna plots")
        
        sub_folder = os.path.join(self.result_folder, 'study_plots', 'optuna_plots')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        self.figure_storage = ccbdl.storages.storages.FigureStorage(
            sub_folder, types=("png", "pdf"))

        figures_list = []
        figures_names = []
    
        fig = optuna.visualization.plot_optimization_history(self.study)
        figures_list.append(fig)
        figures_names.append("optimization_history")
        
        fig = optuna.visualization.plot_contour(
            self.study, params=["learning_rate_exp", "optimizer"])
        figures_list.append(fig)
        figures_names.append("contour")
    
        fig = optuna.visualization.plot_parallel_coordinate(
            self.study, params=["learning_rate_exp", "optimizer"])
        figures_list.append(fig)
        figures_names.append("parallel_coordinate")
    
        fig = optuna.visualization.plot_param_importances(self.study)
        figures_list.append(fig)
        figures_names.append("param_importances")
    
        fig = optuna.visualization.plot_slice(
            self.study, params=["learning_rate_exp", "optimizer"])
        figures_list.append(fig)
        figures_names.append("plot_slice")
    
        # Now use store_multi to store all figures at once
        self.figure_storage.store_multi(figures_list, figures_names)
        
        end_time = datetime.now()
        self.overall_duration = sum(self.durations, timedelta()) + (end_time - start_time)
        
        parameter_storage.write("\nStudy statistics: ")
        parameter_storage.write(
            f"  Number of finished trials: {len(self.study.trials)}")
        parameter_storage.write(
            f"  Number of pruned trials: {len(pruned_trials)}")
        parameter_storage.write(
            f"  Number of complete trials: {len(complete_trials)}")
        parameter_storage.write(
            f"  Time of this entire study: {str(self.overall_duration)[:-7]}")
        parameter_storage.write(
            f"\nBest trial: Nr {self.study.best_trial.number}")
        parameter_storage.write(f"  Best Value: {self.study.best_trial.value}")

        parameter_storage.write("  Params: ")
        for key, value in self.study.best_trial.params.items():
            parameter_storage.write(f"    {key}: {value}")
        parameter_storage.write("\n")      
