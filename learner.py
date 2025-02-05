import torch
from ccbdl.utils import DEVICE
from ccbdl.learning.auto_encoder import BaseAutoEncoderLearning
from ccbdl.utils.logging import get_logger
#from torcheval.metrics import FrechetInceptionDistance
from fid_custom import FrechetInceptionDistance
from utils import load_classifier_rgb
from plots import *
import os


class Learner(BaseAutoEncoderLearning):
    def __init__(self,
                 trial_path: str,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(train_data, test_data, val_data, trial_path, 
                         learner_config, task=task, logging=logging)

        self.model = model
        
        self.device = DEVICE
        print(self.device)
        
        self.criterion_name = learner_config["criterion"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.learner_config = learner_config
        
        self.lr = 10**self.lr_exp

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)
        
        self.optimizer = getattr(torch.optim, self.optimizer)(self.model.parameters(), lr=self.lr)

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path
        
        if self.learner_config["layer"] == 'linear': 
            # Load the classifier
            self.classifier = load_classifier_rgb(layer='linear')
        elif self.learner_config["layer"] == 'nlrl':
            # Load the classifier
            self.classifier = load_classifier_rgb(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
        
        self.plotter.register_default_plot(TimePlot(self))
        self.plotter.register_default_plot(Decode_plot(self))
        self.plotter.register_default_plot(Loss_plot(self))
        # self.plotter.register_default_plot(Attribution_plots_discriminator(self))
        # self.plotter.register_default_plot(Confusion_matrix_gan(self))
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        self.plotter.register_default_plot(Fid_plot(self))  
        # if self.learner_config["en_de_model"] == "en_de":
        #     self.plotter.register_default_plot(Tsne_plot_dis_gan(self))
        # self.plotter.register_default_plot(Psnr_plot(self))
        # self.plotter.register_default_plot(Attribution_plots_classifier(self))
        # self.plotter.register_default_plot(Ssim_plot(self))              
        
        # if self.learner_config["layer"] == "nlrl":
        #     self.plotter.register_default_plot(Hist_plot_discriminator(self))
        #     self.plotter.register_default_plot(Hist_plot_classifier(self))

        self.parameter_storage.store(self)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        
        # self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters of the model: ")
        # self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters of the model: ")
        
        self.fid_metric = FrechetInceptionDistance(model=self.classifier, feature_dim=10, device=self.device)
    
    def _train_epoch(self, train=True):
        self.logger = get_logger()
        self.logger.info("training.")

        self.model.train()

        for i, data in enumerate(self.train_data):
            self.logger.info("starting train batch")
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            
            self.logger.info("encoding decoding the images")
            encoded_images = self._encode(images)
            # print(images.shape)
            # print(self.model.encoder.inc(images).shape)
            # print(self.model.encoder.encoder1(images).shape)
            # print(self.model.encoder.encoder2(images).shape)
            # print(encoded_images.shape)
            decoded_images = self._decode(encoded_images)
            
            self.logger.info("train loss")
            self.train_loss = self.criterion(decoded_images, images)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()
            
            def grayscale_to_rgb(images):
                # `images` is expected to be of shape [batch_size, 1, height, width]
                return images.repeat(1, 3, 1, 1)
            
            # Convert real and fake images from grayscale to RGB
            images_rgb = grayscale_to_rgb(images)
            decoded_images_rgb = grayscale_to_rgb(decoded_images.detach())

            # Update the metric for real images and fake images of RGB
            self.fid_metric.update(images_rgb, is_real=True)
            self.fid_metric.update(decoded_images_rgb, is_real=False)
            
            self.fid = self.fid_metric.compute()


            self.data_storage.store([self.epoch, self.batch, self.train_loss, 
                                     self.test_loss, self.fid])

            if train:
                self.batch += 1    
                if len(self.train_data) - 1:
                    self.data_storage.dump_store("real_images", images)
                    self.data_storage.dump_store("fake_images", decoded_images.detach())
                    self.data_storage.dump_store("labels", labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("real_images_", images)
                    self.data_storage.dump_store("fake_images_", decoded_images.detach())
                    self.data_storage.dump_store("labels_", labels)
    
    def _test_epoch(self):
        self.logger = get_logger()
        self.logger.info("testing.")
    
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                self.logger.info("starting test batch")
    
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).long()
    
                # Encoding Decoding
                self.logger.info("encoding decoding of the test images")
                encoded_images = self._encode(images)
                # print(images.shape)
                # print(self.model.encoder.inc(images).shape)
                # print(self.model.encoder.encoder1(images).shape)
                # print(self.model.encoder.encoder2(images).shape)
                # print(encoded_images.shape)
                decoded_images = self._decode(encoded_images)

                self.logger.info("test loss")
                loss += self.criterion(decoded_images, images)
                
                if len(self.test_data) - 1:
                    self.data_storage.dump_store("real_images_test", images)                   
                    self.data_storage.dump_store("fake_images_test", decoded_images.detach())
                    self.data_storage.dump_store("labels_test", labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("real_images_test_", images)                   
                    self.data_storage.dump_store("fake_images_test_", decoded_images.detach())
                    self.data_storage.dump_store("labels_test_", labels)
        
        self.test_loss = loss / (i+1)
    
    def _validate_epoch(self):
        pass

    def _encode(self, ins):
        return self.model.encoder(ins)

    def _decode(self, ins):
        return self.model.decoder(ins)
    
    def _update_best(self):
        if self.fid < self.best_values["FidScore"]:
            self.best_values["TrainLoss"] = self.train_loss.item()
            self.best_values["TestLoss"] = self.test_loss.item()
            self.best_values["FidScore"] = self.fid.item()
            self.best_values["Batch"] = self.batch

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"TrainLoss":      self.train_loss.item(),
                           "TestLoss":       self.test_loss.item(),
                           "FidScore":       self.fid.item(),
                           "Batch":          self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"TrainLoss":      self.train_loss.item(),
                               "TestLoss":       self.test_loss.item(),
                               "FidScore":       self.fid.item(),
                               "Batch":          self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'TrainLoss': self.train_loss.item(),
                                    "TestLoss":  self.test_loss.item(),
                                    "FidScore":  self.fid.item(),
                                    'model_state_dict': self.model.state_dict()}
        
        self.data_storage.dump_store("epochs_gen", self.epoch)
            
    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'model_state_dict': self.best_state_dict},
                   self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'TrainLoss': self.train_loss.item(),
                    "TestLoss":  self.test_loss.item(),
                    "FidScore":  self.fid.item(),
                    'model_state_dict': self.model.state_dict()},
                   self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))


class Conditional_Learner(Learner):
    def __init__(self,
                 trial_path: str,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(trial_path, model, train_data, test_data, val_data, task, learner_config, logging)
        
        # self.plotter.register_default_plot(Tsne_plot_dis_cgan(self))
        # self.plotter.register_default_plot(Tsne_plot_images_separate(self))
        # self.plotter.register_default_plot(Tsne_plot_images_combined(self))
        # self.plotter.register_default_plot(Confusion_matrix_classifier(self, **{"ticks": torch.arange(0, 10, 1).numpy()}))
    
    def _train_epoch(self, train=True):
        self.logger = get_logger()
        self.logger.info("training.")

        self.model.train()

        for i, data in enumerate(self.train_data):
            self.logger.info("starting train batch")
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            
            self.logger.info("encoding decoding the images")
            encoded_images = self._encode(images, labels)
            # print(images.shape)
            # print(self.model.encoder.inc(images).shape)
            # print(self.model.encoder.encoder1(images).shape)
            # print(self.model.encoder.encoder2(images).shape)
            # print(encoded_images.shape)
            decoded_images = self._decode(encoded_images, labels)
            
            self.logger.info("train loss")
            self.train_loss = self.criterion(decoded_images, images)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()
            
            def grayscale_to_rgb(images):
                # `images` is expected to be of shape [batch_size, 1, height, width]
                return images.repeat(1, 3, 1, 1)
            
            # Convert real and fake images from grayscale to RGB
            images_rgb = grayscale_to_rgb(images)
            decoded_images_rgb = grayscale_to_rgb(decoded_images.detach())

            # Update the metric for real images and fake images of RGB
            self.fid_metric.update(images_rgb, is_real=True)
            self.fid_metric.update(decoded_images_rgb, is_real=False)
            
            self.fid = self.fid_metric.compute()

            self.data_storage.store([self.epoch, self.batch, self.train_loss, 
                                     self.test_loss, self.fid])

            if train:
                self.batch += 1
                if len(self.train_data) - 1:
                    self.data_storage.dump_store("real_images", images)
                    self.data_storage.dump_store("fake_images", decoded_images.detach())
                    self.data_storage.dump_store("labels", labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("real_images_", images)
                    self.data_storage.dump_store("fake_images_", decoded_images.detach())
                    self.data_storage.dump_store("labels_", labels)
    
    def _test_epoch(self):
        self.logger = get_logger()
        self.logger.info("testing.")
    
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                self.logger.info("starting test batch")
    
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).long()
    
                # Encode
                self.logger.info("encoding decoding the test images")
                encoded_images = self._encode(images, labels)
                # print(images.shape)
                # print(self.model.encoder.inc(images).shape)
                # print(self.model.encoder.encoder1(images).shape)
                # print(self.model.encoder.encoder2(images).shape)
                # print(encoded_images.shape)
                decoded_images = self._decode(encoded_images, labels)
                
                self.logger.info("test loss")
                loss += self.criterion(decoded_images, images)
                
                if len(self.test_data) - 1:
                    self.data_storage.dump_store("real_images_test", images)                   
                    self.data_storage.dump_store("fake_images_test", decoded_images.detach())
                    self.data_storage.dump_store("labels_test", labels)
                if self.epoch == self.learner_config["num_epochs"] - 1:
                    self.data_storage.dump_store("real_images_test_", images)                   
                    self.data_storage.dump_store("fake_images_test_", decoded_images.detach())
                    self.data_storage.dump_store("labels_test_", labels)
        
        self.test_loss = loss / (i+1)
    
    def _encode(self, ins, labels):
        return self.model.encoder(ins, labels)
    
    def _decode(self, ins, labels):
        return self.model.decoder(ins, labels)

