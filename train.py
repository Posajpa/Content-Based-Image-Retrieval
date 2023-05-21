import logging
import time
import torch
from pathlib import Path
import yaml
from model import VGG
from model import ResNet
from dataset import get_train_data
from utils import seed_everything
from torchsummary import summary


class TransferLearningTrainer:
    def __init__(self, config):
        self.config = config
        self.cost_function = torch.nn.CrossEntropyLoss()
        self.learn_rate = self.config["learning_rate"]
        # Parameters
        self.max_epochs = config["max_epochs"]
        self.save_checkpoint_every = config["save_checkpoint_every"]
        self.early_stopping_patience = config["early_stopping_patience"]
        self.trigger = 0

    def setup_trainer(self, net, checkpoint_path):
        # Create the checkpoint tree
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.learn_rate)

    def save(self, net, epoch, is_best=False):
        if is_best:
            torch.save(net.state_dict(), self.checkpoint_path / f'best.pth')  # in this case we store only the model
        else:
            save_dict = dict(
                model=net.state_dict(),
                optimizer=self.optimizer.state_dict(),
                # optimizer has parameters as well, you want to save this to be able to go back to this exact stage of training
            )
            torch.save(save_dict, self.checkpoint_path / f'epoch-{epoch}.pth')

    def train(self, net, train_loader, val_loader):

        # For each epoch, train the network and then compute evaluation results
        start = time.time()
        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        print(f"\tStart training...")

        for e in range(self.max_epochs):
            train_loss, train_accuracy = self.train_step(net, train_loader)
            val_loss, val_accuracy = self.val_step(net, val_loader)
            print(
                '\tEpoch {}: Training loss {:.4f}, Training accuracy {:.4f}, Validation loss {:.4f}, '
                'Validation accuracy {:.4f}, Best accuracy {:.4f}'.format(
                    e + 1, train_loss, train_accuracy, val_loss, val_accuracy, best_val_accuracy))
            # # Save the model checkpoints
            # if e % self.save_checkpoint_every == 0 or e == (
            #         self.max_epochs - 1):  # if the current epoch is in the interval, or is the last epoch -> save
            #     torch.save(net.state_dict(), self.checkpoint_path / f'epoch-{e}.pth')

            # Update the best model so far
            if val_accuracy >= best_val_accuracy:
                torch.save(net.state_dict(), self.checkpoint_path / f'best.pth')
                best_val_accuracy = val_accuracy

            # Early Stopping
            if val_loss > best_val_loss:
                self.trigger += 1
                if self.trigger == self.early_stopping_patience:
                    print(
                        f"Validation Accuracy did not improve for {self.early_stopping_patience} epochs. Killing the training...")
                    break
            else:
                # update the best val loss so far
                best_val_loss = val_loss
                self.trigger = 0

        end = time.time()
        print('\tTraining duration: ', end - start)
        logging.basicConfig(filename='logs/logs.txt',
                            format="%(asctime)s %(message)s",
                            level=logging.INFO)
        logging.info(f"Model = {opt.model_name}, epochs = {self.max_epochs}, learning rate = {self.learn_rate}, best train & loss:  {round(best_val_accuracy, 3)}, {round(best_val_loss, 3)}")

    def train_step(self, net, train_loader):
        net.train()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0

        for idx, (data, name, target) in enumerate(train_loader):
            data, target = data.to(config["device"]), target.to(config["device"])
            output = net(data)
            loss = self.cost_function(output, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def val_step(self, net, val_loader):
        net.eval()
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        with torch.no_grad():
            for idx, (data, name, target) in enumerate(val_loader):
                data, target = data.to(config["device"]), target.to(config["device"])
                output = net(data)
                loss = self.cost_function(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == target).sum().item()
                total_samples += target.size(0)

        return total_loss / total_samples, total_correct / total_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Training Parser")
    # Misc
    parser.add_argument("--config_path", required=False, type=str, default="./config/config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--model_name", choices=["VGG16", "ResNet"], required=True, help="Name of the model used")
    parser.add_argument("--checkpoint_path", required=False, type=str, default="./checkpoints",
                        help="path where the checkpoints will be stored")
    opt = parser.parse_args()  # parse the arguments, this creates a dictionary name : value

    # Seed the training
    seed_everything()

    # Load the configuration file
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {opt.config_path}")

    # Get dataset
    train_loader, val_loader, out_layer = get_train_data(config)

    # Create checkpoint path
    checkpoint_path = Path(opt.checkpoint_path) / opt.model_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"\tCheckpoint file created in: {opt.checkpoint_path}")
    print('-----------------------------------------------------')
    # save a copy of the config file being used, to be sure. Append the command line parameters and out_layer
    config.update({'out_layer_size': out_layer})
    config.update({'command_line': vars(opt)})
    with open(checkpoint_path / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create the model
    if opt.model_name == "VGG16":
        model = VGG(config, out_layer)
    if opt.model_name == "ResNet":
        model = ResNet(config, out_layer)
    print(f"\tModel selected: " + opt.model_name)
    summary(model)
    print('-----------------------------------------------------')
    model.to(config["device"])

    trainer = TransferLearningTrainer(config)
    trainer.setup_trainer(model, checkpoint_path)
    trainer.train(model, train_loader, val_loader)
