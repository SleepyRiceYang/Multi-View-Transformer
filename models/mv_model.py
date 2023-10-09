import os
import numpy as np
from tqdm import tqdm
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train import Model as MsModel
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from mindspore.parallel import set_algo_parameters

from datasets.bboxdataset import WLDATASET
from mindspore.dataset import DataLoader

import mindspore.common.dtype as mstype
import mindspore.nn.optim as optim

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from nets.mv_net import net

class Model:
    def __init__(self, config):
        self.config = config
        self._create_dataset()
        self._create_net()
        self._create_log()
        self._create_optimizer()
        self._create_criterion()
        
        if self.config['eval']['ckpt_path'] != None:
            self.load(self.config['eval']['ckpt_path'])
            self._eval_final(0, mode='val')

    def _create_net(self):
        network_params = self.config['network']
        self.device_id = network_params['device']
        mindspore.set_context(device_target='GPU', device_id=self.device_id)
        self.epochs = self.config['optim']['num_epochs']
        model_name = self.config['network']['model_name']
        num_classes = self.config['data']['num_classes']
        in_channels = self.config['data']['in_channels']

        self.views = 9
        from nets.mv_net import Net
        self.net = Net(cube_size=32, config=self.config, 
                       shape=self.config['data']['bbox_size'])
        self.net = self.net.to(self.device)

    def _create_log(self):
        self.best_result = {"test": {"epoch":0, "acc": 0, "auc": 0},
                            "val": {"epoch":0, "acc": 0, "auc": 0},
                            "train": {"epoch":0, "acc": 0, "auc": 0}}
        
        model_name = self.config['network']['model_name']
        model_suffix = self.config['network']['model_suffix']
        seed = "_"+str(self.config['network']['seed'])
        logging_params = self.config['logging']
        timestamp = time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime())
        exp_des = self.config['network']['description']

        self.ckpt_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp+'_'+exp_des, 'ckpt')
        self.tb_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp+'_'+exp_des, 'tensorboard')
        self.log_path = os.path.join(
            'results', model_name, model_suffix+seed+timestamp+'_'+exp_des, 'log')
        
        if logging_params['use_logging']:
            from utils.log import get_logger
            from utils.parse import format_config
            
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = get_logger(self.log_path)
            
            self.logger.info(">>>The config is:")
            self.logger.info(format_config(self.config))

def _create_dataset(self):
    def _init_fn(worker_id):
        """Workers init func for setting random seed."""
        np.random.seed(self.config['network']['seed'])
        # random.seed(self.config['network']['seed'])
        data_params = self.config['data']
        # making training dataset and dataloader
        train_params = self.config['train']
        train_trans_seq = self._resolve_transforms(train_params['aug_trans'])
        train_dataset = WLDATASET(phase="train", 
                            task=self.config["task"],
                            pkl_file=os.path.join(data_params["data_root"],data_params["train_file"]),
                            transforms=train_trans_seq,
                            bbox_path=data_params["bbox_path"],
                            bbox_size=data_params['bbox_size'],
                            lung_crop=data_params["lung_crop"]
                            )

        train_dataset = mindspore.dataset.GeneratorDataset(train_dataset, ["image", "label"])
        train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
        train_dataset = train_dataset.batch(batch_size=train_params['batch_size'], drop_remainder=True)
   
        train_dataset = train_dataset.repeat(count=1)
    
        train_dataset = train_dataset.device_que(prefetch_size=train_params['pin_memory'])
        self.train_loader = train_dataset

        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = WLDATASET(phase="eval", 
                            task=self.config["task"],
                            pkl_file=os.path.join(data_params["data_root"],data_params["val_file"]),
                            transforms=eval_trans_seq,
                            bbox_path=data_params["bbox_path"],
                            bbox_size=data_params['bbox_size'],
                            lung_crop=data_params["lung_crop"])
    
        eval_dataset = mindspore.dataset.GeneratorDataset(eval_dataset, ["image", "label"])
        eval_dataset_ms = eval_dataset_ms.batch(batch_size=eval_params['batch_size'], drop_remainder=False)
    
        eval_dataset_ms = eval_dataset_ms.device_que(prefetch_size=eval_params['pin_memory'])
        self.val_loader = eval_dataset_ms

        test_dataset = WLDATASET(phase="test", 
                            task=self.config["task"],
                            pkl_file=os.path.join(data_params["data_root"],data_params["test_file"]),
                            transforms=eval_trans_seq,
                            bbox_path=data_params["bbox_path"],
                            bbox_size=data_params['bbox_size'],
                            lung_crop=data_params["lung_crop"])
    
        test_dataset = mindspore.dataset.GeneratorDataset(test_dataset, ["image", "label"])
    
        test_dataset = test_dataset.batch(batch_size=eval_params['batch_size'], 
                                                drop_remainder=False)
        test_dataset = test_dataset.device_que(prefetch_size=eval_params['pin_memory'])
        self.test_loader = test_dataset          
    
    def _create_optimizer(self):
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(self.net.trainable_params(),
                                learning_rate=sgd_params['base_lr'],
                                momentum=sgd_params['momentum'],
                                weight_decay=sgd_params['weight_decay'],
                                nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(self.net.trainable_params(),
                                learning_rate=adam_params['base_lr'],
                                beta1=adam_params['betas'][0],
                                beta2=adam_params['betas'][1],
                                weight_decay=adam_params['weight_decay'],
                                eps=adam_params['eps'])
        elif optim_params['optim_method'] == 'adadelta':
            adadelta_params = optim_params['adadelta']
            optimizer = optim.Adadelta(self.net.trainable_params(),
                                    learning_rate=adadelta_params['base_lr'],
                                    weight_decay=adadelta_params['weight_decay'],
                                    eps=adadelta_params['eps'])

        # choosing whether to use lr_decay and related parameters
        if optim_params['use_lr_decay']:
            # 使用MindSpore的learning_rate_schedule类来创建学习率衰减策略
            import mindspore.nn.learning_rate_schedule as lr_schedule
            if optim_params['lr_decay_method'] == 'cosine':
                lr_scheduler = lr_schedule.cosine_decay(
                                learning_rate=optimizer.learning_rate,
                                end_lr=0,
                                max_epoch=self.config['optim']['num_epochs'])
            if optim_params['lr_decay_method'] == 'lambda':
                def lr_lambda(epoch): return (1 - float(epoch) /
                                            self.config['optim']['num_epochs'])**0.9
                lr_scheduler = lr_schedule.LambdaLR(
                    learning_rate=optimizer.learning_rate,
                    lr_lambda=lr_lambda)
            # 使用MindSpore的update_cell类来更新优化器的学习率
            from mindspore.nn.wrap.cell_wrapper import WithLossCell, TrainOneStepCell, TrainOneStepWithLossScaleCell
            optimizer = optim.update_cell(optimizer, lr_scheduler)
        self.optimizer = optimizer

    def _create_criterion(self):
        self.egfr_ins_loss = GatedSmoothLoss(self.config, use_cent_loss=True)
        self.egfr_loss = GatedSmoothLoss(self.config, use_cent_loss=True)

    def run(self):
        self.config['optim']['num_epochs'] = 150
        for epoch_id in range(self.config['optim']['num_epochs']):
            self._train(epoch_id)
            if self.config['optim']['use_lr_decay']:
                self.lr_scheduler.step()
            self._eval(epoch_id, "val")

    def _train(self, epoch_id):
        self.net.train()
        scaler = GradScaler()
        
        with tqdm(total=len(self.train_loader)) as pbar:
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                pre_ins, fea_ins ,predictions, features = self.net(inputs) 
                
                targets_ins = targets.unsqueeze(1).repeat(1,self.views).view(-1)
            
                loss_ins = self.egfr_ins_loss(pre_ins, targets_ins, fea_ins)
                loss_bag = self.egfr_loss(predictions, targets, features)
                loss = loss_ins + loss_bag
                loss.backward()
                
                self.optimizer.step()
                
                pbar.update(1) 

    def _forward(self, data_loader):
        net = self.net
        net.eval()

        total_loss = 0
        total_bag_output = []
        total_bag_target = []
        num_steps = data_loader.__len__()
        
        with torch.no_grad():
            with tqdm(total=num_steps) as pbar:
               for inputs, targets in data_loader:
                   inputs = inputs.to(self.device)
                   targets = targets.to(self.device)
                   pre_ins, fea_ins, predictions, features = self.net(inputs)
                    
                   tar_ins  = targets.unsqueeze(1).repeat(1, self.views).view(-1)
                   loss_ins = self.egfr_ins_loss(pre_ins, tar_ins, fea_ins)
                   egfr_loss = self.egfr_loss(predictions, targets, features)
                   total_loss += (egfr_loss.item()+loss_ins.item())
                   
                   total_bag_output.extend(predictions.data.cpu().numpy())
                   total_bag_target.extend(targets.data.cpu().numpy())

                   pbar.update(1) 
        return total_bag_output, total_bag_target, total_loss

    def _eval_final(self, epoch, mode="val"):
        logger = self.logger
        if mode == "val":
            data_loader = self.val_loader
        elif mode == "train":
            data_loader = self.train_loader
        elif mode == "test":
            data_loader = self.test_loader
        
        pre_bag, tar_bag, loss= self._forward(data_loader) 
        
        from models.metrics import test_model
        
        pre_bag = np.array(pre_bag)
        tar_bag = np.array(tar_bag)
        
        #print(pre_bag.shape, tar_bag.shape)
          
        pre_pos_bag = pre_bag[:,1]
        test_model(pre_pos_bag, tar_bag)
        sys.exit()

    def _eval(self, epoch, mode="val"):
        logger = self.logger
        if mode == "val":
            data_loader = self.val_loader
        elif mode == "train":
            data_loader = self.train_loader
        elif mode == "test":
            data_loader = self.test_loader
        
        pre_bag, tar_bag, loss= self._forward(data_loader) 
        
        from models.metrics import test_model
        
        pre_bag = np.array(pre_bag)
        tar_bag = np.array(tar_bag)
        
        confusion_matrix = metrics.get_confusion_matrix(pre_bag, tar_bag)
        num_correct = sum(np.argmax(pre_bag, 1) == tar_bag)
        
        acc = num_correct / len(tar_bag)
        loss = loss / len(tar_bag)

        confusion_matrix_per = confusion_matrix.copy()
        for i in range(confusion_matrix.shape[0]):
            confusion_matrix_per[i] /= sum(confusion_matrix_per[i])

        recall = metrics.get_recall(confusion_matrix)
        precision = metrics.get_precision(confusion_matrix)
        TN,FP,FN,TP = confusion_matrix.flatten()
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        

        if self.config['data']['num_classes'] > 1:
            macro_auc = metrics.get_mul_cls_auc(pre_bag, tar_bag)
            each_auc = metrics.get_each_auc(pre_bag, tar_bag)

            logger.info("[{}] Epoch:{}\n confusion matrix:\n{}\n percentage confusion matrix:\n{} recall:{}\n percision: {}\n acc:{}/{}={:.5}\n Sensitivity:{:.5} \nSpecificity:{:.5}\n PPV:{:.5}\n NPV:{:.5}\n loss:{:.5}".format(
                mode, epoch, confusion_matrix, 
                confusion_matrix_per * 100,
                recall * 100,
                precision * 100,
                #each_auc * 100,
                #macro_auc * 100,
                num_correct, len(tar_bag), 
                acc, Sensitivity, 
                Specificity,
                PPV, NPV, loss))
        else:
            raise ValueError(f"num_classes = {self.config['data'][num_classes]}")
        self.writer.add_scalar('%s_loss' % mode, loss, epoch)
        self.writer.add_scalar('%s_auc' % mode, macro_auc, epoch)
        results = self.best_result[mode]

        if macro_auc > results['auc']:
            results['auc'] = macro_auc
            results['epoch'] = epoch
            logger.info('[Info] Epochs:%d, %s AUC improve to %g' % 
                    (epoch, mode, macro_auc))
            snapshot_name = '%s_epoch_%d_loss_%.5f_auc_%.5f_lr_%.10f' % \
            (mode, epoch, loss, macro_auc, self.optimizer.param_groups[0]['lr'])
            
            # save_best_auc
            torch.save(self.net.state_dict(), os.path.join(
                self.ckpt_path, snapshot_name + '.pth'))
        else:
            logger.info("[Info] Epoch:%d, %s AUC didn't improve, current best AUC is %g, epoch:%g" % (epoch, mode, results['auc'], results['epoch']))

        return pre_bag, tar_bag

    from mindspore.train.serialization import load_checkpoint, load_param_into_net
    def load(self, ckpt_path):
        param_dict = load_checkpoint(ckpt_path)
    
        if self.config['network']['use_parallel']:
            load_param_into_net(self.net, param_dict)
        else:
            load_param_into_net(self.net, param_dict)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    from mindspore import save_checkpoint
    def save(self, epoch):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()

        checkpoint_name = os.path.join(self.ckpt_path, f'{epoch}.ckpt')
        save_checkpoint(state_dict, checkpoint_name)

    def _resolve_transforms(self, aug_trans_params):
        """
            According to the given parameters, resolving transform methods
        :param aug_trans_params: the json of transform methods used
        :return: the list of augment transform methods
        """
        trans_seq = []
        for trans_name in aug_trans_params['trans_seq']:
            if trans_name == 'center_crop':
                resize_params = aug_trans_params['center_crop']
                trans_seq.append(extend_transforms.CenterCrop(resize_params['size']))
            elif trans_name == 'wc_ww':
                wc_ww_params = aug_trans_params["wc_ww"]
                trans_seq.append(extend_transforms.WC_WW(wc_ww_params['wc'], wc_ww_params['ww']))
            elif trans_name == 'to_tensor':
                trans_seq.append(extend_transforms.ToTensor())
            else:
                raise NotImplementedError("trans_name: "+trans_name)
        return trans_seq
    
    def inference(self):
        best_ckpt_path = get_best_ckpt(self.ckpt_path)
        self.load(best_ckpt_path)
        for mode in ["Training","Validation", "Test"]:
            if mode == "Training":
                data_loader = train_loader
            elif mode == "Validation":
                data_loader = val_loader
            elif mode == "Test":
                data_loader = test_loader
        output, target, loss = self._forward(data_loader)
        target_dir = os.path.join(
                self.config["inference"]["npy_dir"],
                self.config['network']['model_name'], 
                self.ckpt_path.split("/")[-2])
        
        file_dir=os.path.join(target_dir,"npy")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(os.path.join(file_dir,"{}_output.npy".format(mode)), output)
        np.save(os.path.join(file_dir,"{}_target.npy".format(mode)), target)
        

class GatedSmoothLoss(nn.Cell):
    def __init__(self, config, use_cent_loss=True):
        super(GatedSmoothLoss, self).__init__()
        self.config = config
        self.use_cent_loss = use_cent_loss
        self.beta = 1

        from models.center_loss import CenterLoss 
        self.center_loss = CenterLoss(num_classes=config['data']['num_classes'], feat_dim=16, use_gpu=True)
        self.cross_entropy_loss = SmoothCrossEntropyWithMask()

    def construct(self, predictions, targets, features):
        clas_loss = self.cross_entropy_loss(predictions, targets)

        if self.use_cent_loss:
            cent_loss = self.center_loss(features, targets)
            print(f'center_loss:{cent_loss.asnumpy():.2f}')
            return clas_loss + self.beta * cent_loss
        else:
            return clas_loss

class SmoothCrossEntropyWithMask(nn.Cell):
    def __init__(self, epsilon=0.1):
        super(SmoothCrossEntropyWithMask, self).__init__()
        self.epsilon = epsilon
        self.ops = ops.Operations()

    def construct(self, predictions, targets):
        predictions = predictions[:, 1]
        bc = predictions.shape[0]
        
        y_i_hat = self.epsilon * (1 - targets) + (1 - self.epsilon) * targets
        log_p = self.ops.log(predictions)
        log_1_p = self.ops.log(1 - predictions)
        
        w_pos = 0.6
        w_neg = 0.4
        
        loss_nll = -(w_pos * y_i_hat * log_p + w_neg * (1 - y_i_hat) * log_1_p)
        
        mean_loss = self.ops.mean(loss_nll)
        
        print(f'cross_entropy_loss:{mean_loss.asnumpy():.5f}')
        return mean_loss

