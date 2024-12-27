import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve


"""
    Classes:
        - ModelInspector;
        - MetricTracker;
        - ConvolutionalBlock;
        - FeatureExtraction;
        - FullyConnected;
        - Classification;
        - DartVetterFiscale2024.
"""

class ModelInspector:
  def __init__(self, model):
    self.__model = model

  def count_trainable_params(self):
    return sum(p.numel() for p in self.__model.parameters() if p.requires_grad)

  def print_layer_params(self):
    print(f"{'Layer':<30} {'Param Count':<20}")
    print("="*50)
    for name, param in self.__model.named_parameters():
      print(f"{name:<30} {param.numel():<20}")


class MetricTracker:
  def __init__(self, device, num_classes):
    self.__losses = []
    self.__accuracies = []
    self.__average = 'macro'
    self.__task = ''
    self.__num_classes = num_classes
    if num_classes == 1:
      self.__task = 'binary'
    else:
      self.__task = 'multiclass'
    
    self.__precision_metric = torchmetrics.Precision(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__recall_metric = torchmetrics.Recall(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__f1_metric = torchmetrics.F1Score(task=self.__task, num_classes=self.__num_classes, average=self.__average).to(device)
    self.__auc_roc_metric = torchmetrics.AUROC(task=self.__task, num_classes=self.__num_classes).to(device)
    self.__precision_recall_curve_metric = torchmetrics.PrecisionRecallCurve(task=self.__task, num_classes=self.__num_classes).to(device)  # return a tensor containing three lists: recall, precision, thresholds


  def _compute(self):
    """
      Compute evaluation metrics during model training
    """
    precision               = self.__precision_metric.compute()
    recall                  = self.__recall_metric.compute()
    f1                      = self.__f1_metric.compute()
    auc_roc                 = self.__auc_roc_metric.compute()
    precision_recall_curve  = self.__precision_recall_curve_metric.compute()
    # Reset metrics for next computation
    self.__precision_metric.reset()
    self.__recall_metric.reset()
    self.__f1_metric.reset()
    self.__auc_roc_metric.reset()
    self.__precision_recall_curve_metric.reset()

    return precision.item(), recall.item(), f1.item(), auc_roc.item(), precision_recall_curve


  def _update(self, outputs, labels, loss):
    """
      This method compute and update the evaluation metrics at the end of each training epoch.
      Input:
        - outputs: the predictions of the model. type: torch.Tensor. shape: [batch_size, num_classes];
        - labels: the true labels of the examples. type: torch.Tensor. shape: [batch_size];
        - loss: loss function value. type: torch.Tensor.
      Further information:
        About _precision_recall_curve_metric:
        Since outputs.shape:[batch_size, num_classes] and labels.shape:[batch_size] you have to remove the extra dimension 
        from outputs (or add a dimension to labels). Moreover, torchmetrics.PrecisionRecallCurve expects the 
        labels tensor contains integer values (typically torch.int or torch.long). To be sure labels is of the
        correct data type (in some cases it could be torch.float32 with i-th label value = 1.0), we cast the tensor.
    """
    self.__losses.append(loss.item())

    if self.__num_classes == 1:
      # Binary classification
      preds = torch.round(torch.sigmoid(outputs).squeeze(-1)) # Remove the extra dimension from outputs tensor so that it has the same shape of labels. Store values in preds.
      accuracy = (preds == labels.unsqueeze(1)).float().mean()
    else:
      # Multi-class classification
      # Computing accuracy for multi-class classification requires the following steps:
      # 1. Use torch.argmax on model's predictions to achieve the index of the class with highest likelihood for each sample;
      # 2. Compare y_pred with y_true
      preds = torch.argmax(outputs, dim=1)
      accuracy = (preds == labels).float().mean()

    self.__accuracies.append(accuracy.item())
    self.__precision_metric(preds, labels)
    self.__recall_metric(preds, labels)
    self.__f1_metric(preds, labels)
    self.__auc_roc_metric(outputs, labels)    # AUC-ROC and PR curve require the logits or probabilities, then use the tensor outputs
    self.__precision_recall_curve_metric(outputs.squeeze(dim=1), labels.long())  

  def _plot(self, x, y, xlabel, ylabel, title, path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='.')
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=1200)
    plt.close()

  def _plot_training_metrics(self, epoch, epoch_loss, precision, recall, f1, auc_roc, path):
    """
      Plot the values for each evaluation metrics during training
      Input:
        - epoch_loss, precision, recall, f1, auc_roc. type: list
    """
    self._plot(epoch, epoch_loss, 'Epoch', 'Epoch loss', 'Loss function', path + 'training_epoch.png')
    self._plot(epoch, precision, 'Epoch', 'Precision', 'Training precision', path+'training_precision.png')
    self._plot(epoch, recall, 'Epoch', 'Recall', 'Training recall', path+'training_recall.png')
    self._plot(epoch, f1, 'Epoch', 'F1-score', 'Training f1-score', path+'training_f1.png')
    self._plot(epoch, auc_roc, 'Epoch', 'AUROC', 'Training Area under ROC', path+'training_aucroc.png')


  def compute_evaluation_metrics(self, all_preds, all_probs, all_labels):
    # Metrics computation with scikit-learn
    #NOTE. Experimental: weighted average during evaluation
    self.__average = 'binary' #weighted allows you to calculate metrics for each label and find their average weighted by support (the number of true instancs for each label). Account for class imbalance. This can result in an F-score that is not between precision and recall.

    # Precision, recall and F1 score
    try:
      precision = precision_score(all_labels, all_preds, average=self.__average)  # NOTE. prima era 'binary' per binary classification. Check for errors
      recall = recall_score(all_labels, all_preds, average=self.__average)
      f1 = f1_score(all_labels, all_preds, average=self.__average)
    except Exception as e0:
      print(f'In m1_class.py --> Class MetricTracker --> compute_evaluation_metrics() --> Error while computing precision, recall and f-score.\nError:{e0}')

    # Area under ROC
    if self.__num_classes == 1:
      auc_roc = roc_auc_score(all_labels, all_preds)
    else:
      # In multi-class classification this metric requires the probabilities score the model returned for each class. For this reason, we pass all_probs.
      # 'ovr' stands for One-vs-rest. Computes the AUC of each class against the rest. This treats the multiclass case in the same way as the multilabel case. 
      # Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
      try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=self.__average)
        print(f'AUROC: {auc_roc:.4f}')  # su riga separata rispetto a precision, recall ed fscore, altrimenti puoi avere errore UnboundLocalError: local variable 'auc_roc' referenced before assignment nel caso in cui il blocco try{} di auc_roc va in errore.
      except Exception as e1:
        print(f'In m1_class.py --> Class MetricTracker --> compute_evaluation_metrics() --> Error while plotting ROC curve in task:multiclass.\nError:{e1}')

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # Print metrics
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print('Confusion matrix')
    print(f'np.array({np.array(conf_matrix)})')


  def plot_prcurve(self, all_labels, all_preds, path):
    # NOTE. Make this code flexible to handle multi-class classification problems
    # Compute and plot the precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    if path:
      plt.savefig(path+'prcurve.png', dpi=1200)
    else:
      plt.show()
    plt.close()
  
  def plot_roc(self, all_labels, all_preds, path):
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    self._plot(fpr, tpr, 'False positive rate', 'True positive rate', 'ROC curve', path+'roc_curve.png')

  def print_metrics(self):
    """
      Print evaluation metrics. 
      #NOTE. Actually, you don't use this method anywhere
    """
    precision, recall, f1, auc_roc, precision_recall_curve = self.__compute()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC-ROC: {auc_roc}")
    
  def get_precision(self):
    return self.__precision_metric.compute()

  def get_recall(self):
    return self.__recall_metric.compute()

  def get_f1(self):
    return self.__f1_metric.compute()

  def get_auc_roc(self):
    return self.__auc_roc_metric.compute()

  def get_precision_recall_curve(self):
    return self.__precision_recall_curve_metric.compute()

  def get_losses(self):
    return self.__losses

  def get_accuracies(self):
    return self.__accuracies


class ConvolutionalBlock(nn.Module):
  def __init__(self, cfs, cbff, psz, pst, rho, i):
    # Constructor
    super(ConvolutionalBlock, self).__init__()
    # Define private attributes
    self.__block_index = i
    self.__conv_filter_size = cfs
    self.__conv_block_filter_factor = cbff
    self.__pooling_size = psz
    self.__pooling_stride = pst
    # Create layers
    self.__convs = nn.ModuleList()
    self.__dropout = nn.ModuleList()
    self.__batch_norm = nn.ModuleList()
    # Define input-output dimension of the feature maps
    in_channels = 1 if i == 0 else pow(self.__conv_block_filter_factor, 4 + (i - 1))
    out_channels = pow(self.__conv_block_filter_factor, 4 + i)
    # Initialize layers with the input values
    self.__convs.append(nn.Conv1d(in_channels, out_channels, self.__conv_filter_size, padding='same'))
    self.__dropout.append(nn.Dropout(rho))
    self.__batch_norm.append(nn.BatchNorm1d(out_channels))

  def forward(self, x):
    if self.__block_index == 0:
      gv_length = 201
      x = x.view(-1, 1, gv_length)
    x = self.__convs[0](x)
    x = F.relu(x)
    x = self.__dropout[0](x)
    x = F.max_pool1d(x, kernel_size=self.__pooling_size, stride=self.__pooling_stride)
    x = self.__batch_norm[0](x)

    return x


class FeatureExtraction(nn.Module):
  def __init__(self, cbn, cfs, cbff, psz, pst, rho):
    super().__init__()
    # Define the number of convolutional blocks and the container of these blocks
    self.__conv_blocks_num = cbn
    self.__conv_blocks = nn.ModuleList()
    self.__flatten = nn.Flatten()
    self.__output_size = -1
    # Stack the convolutional blocks
    for i in range(self.__conv_blocks_num):
      self.__conv_blocks.append(ConvolutionalBlock(cfs, cbff, psz, pst, rho, i))

  def forward(self, x):
    for i in range(self.__conv_blocks_num):
      x = self.__conv_blocks[i](x)
    output = self.__flatten(x)
    return output

  def get_output_size(self):
    output = self.forward(torch.rand(1, 201)).size()[1] # type: torch.Tensor([1, 768]). By doing so, you access to 768
    return output

  def print_branch(self):
    print(self.__conv_blocks)


class FullyConnected(nn.Module):
  def __init__(self, input_size, fc_units, output_size, rho):
    super(FullyConnected, self).__init__()
    self.__fc_units = fc_units
    self.__fc_layer = nn.Linear(input_size, self.__fc_units)
    self.__dropout = nn.Dropout(rho)
    self.__output_layer = nn.Linear(self.__fc_units, output_size)

  def forward(self, x):
    x = self.__fc_layer(x)
    x = F.relu(x)
    x = self.__dropout(x)
    x = self.__output_layer(x)
    return x


class Classification(nn.Module):
  def __init__(self, input_size, fc_layers_num, fc_units, output_size, rho):
    """
      To test the functionality of the class:
      >>> classification = Classification(768, 4, 1024, 5, 0.3)
      >>> classification.print_branch()
    """
    super(Classification, self).__init__()
    self.__fc_layers_num = fc_layers_num
    self.__fc_blocks = nn.ModuleList()
    if self.__fc_layers_num == 1:
      # Single fully-connected layer which shape is (input x output):(fc_units x output_size)
      self.__fc_blocks.append(FullyConnected(input_size, fc_units, output_size, rho))
    else:
      # print(f"The number of fully connected layers is {self.__fc_layers_num}.")
      # Multiple fully-connected layers. Iterate over the number of layers-1 to generate layers of shape (fc_units x fc_units)
      for i in range(self.__fc_layers_num - 1):
        # print(f"Creating {i}-th layer with {fc_units} x {fc_units} neurons.")
        if i == 0:
          # When you have >2 layers, the first layer is the only one which input size has to be the output size of the feature extraction branch
          self.__fc_blocks.append(FullyConnected(input_size, fc_units, fc_units, rho))
        else:
          self.__fc_blocks.append(FullyConnected(fc_units, fc_units, fc_units, rho))
      # Last layer has fc_units x output_size connections
      # print(f"Creating last layer with {fc_units} x {output_size} neurons.")
      self.__fc_blocks.append(FullyConnected(fc_units, fc_units, output_size, rho))

  def forward(self, x):
    for i in range(self.__fc_layers_num):
      x = self.__fc_blocks[i](x)
    return x

  def print_branch(self):
    print(self.__fc_blocks)


# Definizione di DartVetterFiscale2024, che combina i branch di Feature Extraction e Classification
class DartVetterFiscale2024(nn.Module):
  def __init__(self, cbn, cfs, cbff, psz, pst, rho, fc_layers_num, fc_units, fc_output_size):
    """
      Definition of input variables:
        - cbn: convolutional blocks number;
        - cfs: convolutional filters size;
        - cbff: convolutional block filter factor. It is the filter growth factor, typically its value is 2;
        - psz: pooling size;
        - pst: pooling stride;
        - rho: dropout probability rate. Should be within [0.2, 0.5] (Srivastava et al. 2014);
        - fc_layers_num: number of fully-connected layers;
        - fc_units: number of neurons for each fully-connected layer;
        - fc_output_size: number of output neurons.
    """
    super(DartVetterFiscale2024, self).__init__()
    # Initialize the feature extraction and classification branches
    self.__feature_extraction = FeatureExtraction(cbn, cfs, cbff, psz, pst, rho)
    self.__classification = Classification(self.__feature_extraction.get_output_size(), fc_layers_num, fc_units, fc_output_size, rho)
    # Initialize model's parameters
    self.__initializer_pytorch()

  def __initializer_pytorch(self):
    for m in self.modules():
      if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, x):
    # Stack the two branches
    x = self.__feature_extraction.forward(x)
    x = self.__classification.forward(x)
    return x

  def get_feature_extraction_output(self, input):
    return self.__feature_extraction.forward(input)

  def get_feature_extraction_output_size(self):
    return self.__feature_extraction.get_output_size()

  def get_feature_extraction_number_params(self):
    self.__model_inspector = ModelInspector(self.__feature_extraction)
    return self.__model_inspector.count_trainable_params()

  def get_classification_output(self, input):
    return self.__classification.forward(input)

  def get_fully_connected_number_params(self):
    self.__model_inspector = ModelInspector(self.__classification)
    return self.__model_inspector.count_trainable_params()

  def get_model_params(self):
    self.__model_inspector = ModelInspector(self)
    return self.__model_inspector.count_trainable_params()

  def get_layers_params(self):
    self.__model_inspector = ModelInspector(self)
    return self.__model_inspector.print_layer_params()

