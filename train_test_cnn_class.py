import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pandas import read_csv
import yaml
import m1_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

from dataclasses import dataclass

@dataclass
class InputVariables:
    # Path to working directory
    _local_path: str
    # Datasets
    _training_set_name: str
    _training_set_path: str
    _test_set_path: str
    # Model hyperparameters
    _rateval: float
    _learning_rate: float
    _batch_size: int
    _num_epochs: int
    _num_classes: int
    # Extracted features information
    _save_features: bool
    _features_filename: str
    # Model architecture
    _cbn: int
    _cfs: int
    _cbff: int
    _psz: int
    _pst: int
    _fc_layers_num: int
    _fc_units: int
    _fc_output_size: int
    # Define plots filenames
    _plot_dir: str
    _prcurve_filename: str
    _roc_curve_filename: str

    @classmethod
    def get_input_hyperparameters(cls, filename):
        """
            cls Viene usato nei metodi di classe (decorati con @classmethod) per fare riferimento alla classe stessa, 
            non a una singola istanza. cls permette di accedere a variabili di classe o di creare nuove istanze 
            della classe.
            In questo caso, cls viene usato per creare una nuova istanza di InputVariables all'interno del metodo
            di classe get_input_variables.
        """
        # Get input variables from the config.yaml file and store these values in an InputVariables object 
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            _local_path=config['local_path'],
            _training_set_name=config['training_set_name'],
            _training_set_path=config['training_set_path'],
            _test_set_path=config['test_set_path'],
            _rateval=config['rateval'],
            _learning_rate=config['learning_rate'],
            _batch_size=config['batch_size'],
            _num_epochs=config['num_epochs'],
            _num_classes=config['num_classes'],
            _save_features=config['save_features'],
            _features_filename=config['features_filename'],
            _cbn=config['cbn'],
            _cfs=config['cfs'],
            _cbff=config['cbff'],
            _psz=config['psz'],
            _pst=config['pst'],
            _fc_layers_num=config['fc_layers_num'],
            _fc_units=config['fc_units'],
            _fc_output_size=config['fc_output_size'],
            _plot_dir=config['plot_dir'],
            _prcurve_filename=config['prcurve_filename'],
            _roc_curve_filename=config['roc_curve_filename']
            )
    
    def get_training_set_path(self):
      return self._training_set_path
    
    def get_test_set_path(self):
      return self._test_set_path

    def get_learning_rate(self):
      return self._learning_rate
    
    def get_batch_size(self):
      return self._batch_size

    def get_num_epochs(self):
      return self._num_epochs
    
    def get_num_classes(self):
      return self._num_classes
    
    def get_save_features(self):
      return self._save_features
    
    def get_features_filename(self):
      return self._features_filename

    def get_cbn(self):
        return self._cbn

    def get_cfs(self):
        return self._cfs

    def get_cbff(self):
        return self._cbff

    def get_psz(self):
        return self._psz

    def get_pst(self):
        return self._pst
    
    def get_rho(self):
        return self._rateval

    def get_fc_layers_num(self):
        return self._fc_layers_num

    def get_fc_units(self):
        return self._fc_units

    def get_fc_output_size(self):
        return self._fc_output_size

class Model:
  def __init__(self, model, hyperparameters_object):
    # Initialize attributes
    self.__model = model
    self.__model.to(device)
    self.__hyperparameters = hyperparameters_object
    self.__optimizer = optim.Adam(self.__model.parameters(), lr=self.__hyperparameters.get_learning_rate())
    self.__metric_tracker = m1_class.MetricTracker(device, self.__hyperparameters._num_classes)
    if self.__hyperparameters.get_num_classes() > 1:
      self.__criterion = nn.CrossEntropyLoss()  # multi-class classification
    else:
      self.__criterion = nn.BCEWithLogitsLoss() # binary classification

    if self.__hyperparameters.get_save_features():
      # Create these data structures if and only if features vectors have to be stored
      self.__features_filename = self.__hyperparameters.get_features_filename()
      self.__extracted_features = []  # List to store extracted features
      self.__extracted_labels = []    # List to store the labels associated to the TCEs' extracted features

    # TODO. Insert assert to deal with exceptions (also in m1_class.py about model's architecture)
    # assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"

  def __get_samples_labels(self, path):
    """
      Input:
        - path: path to the TCE csv file.
      Output:
        - samples, labels: two numpy.ndarray containing the TCEs' global views and dispositions, respectively.
      #NOTE. Servirà generalizzare l'else condition per gestire differenti datasets categorici. Potrebbe essere utile
      inserire informazioni di input riguardo il mapping da adottare.
    """
    if self.__hyperparameters.get_num_classes() == 1:
      # Binary classification
      values = np.genfromtxt(path, delimiter=',')
      samples = values[:,:-1]
      labels =  values[:,-1]
    else:
      # Multi-class classification
      df = read_csv(path, header=None)
      # For each row, store the first 201 elements in samples; the last element will be stored in labels
      samples = df.iloc[:, :-1].values  # All columns except the last one
      categorical_labels = df.iloc[:, -1].values
      mapping = {'B': 0, 'E': 1, 'J': 2}
      labels = np.vectorize(mapping.get)(categorical_labels)

    return samples, labels

  def __load_dataset(self, dataset_path, mode='train'):
    # Load input data
    x_train, y_train = self.__get_samples_labels(dataset_path)
    # Convert numpy.ndarray data into torch.Tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long) #torch.long
    # Create DataLoader object for training/testing the model
    train_dataset = TensorDataset(x_train, y_train)
    if mode == 'train':
      shuffle = True
    else:
      shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=self.__hyperparameters.get_batch_size(), shuffle=shuffle)

    return train_loader, y_train

  def load_trained_model(self, model_path):
    """
      Ho creato questa funzione perché magari mi sarà utile in futuro se voglio solo testare senza addestrare
    """
    return self.__model.load_state_dict(torch.load(model_path))

  def train(self, dataset_path):
    # Load data
    train_loader, y_train = self.__load_dataset(dataset_path,'train')

    # Initialize data structures for store evaluation metrics during training
    training_epoch=[]
    training_epoch_loss=[]
    training_precision=[]
    training_recall=[]
    training_f1=[]
    training_auc_roc=[]

    # Compute class weighting for binary classification
    if self.__hyperparameters.get_num_classes() == 1:
      num_pos = y_train.sum()             # Number of positive samples (class 1)
      num_neg = len(y_train) - num_pos    # Number of negative samples(class 0)
      pos_weight = (num_neg / num_pos).clone().detach().to(device)    # Class weighting: inverse class frequency
      # Update loss function based on class weighting
      self.__criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
      # Compute class frequency
      class_counts = torch.bincount(y_train, minlength=self.__hyperparameters.get_num_classes())
      # Inverse class frequency
      class_weights = len(y_train) / (self.__hyperparameters.get_num_classes() * class_counts)
      # Convert into tensor
      class_weights = class_weights.float()
      self.__criterion == nn.BCEWithLogitsLoss(weight=class_weights)
      print(f'Class weights: {class_weights}\n')

    for epoch in range(self.__hyperparameters.get_num_epochs()):
      self.__model.train()
      running_loss = 0.0

      for inputs, labels in train_loader:
        # inputs and labels are:<class 'torch.Tensor'>. inputs shape is [n_batch, 201], labels shape is [n_batch]
        # Transfer data to the device of computation
        inputs, labels = inputs.to(device), labels.to(device)
        # Initialize gradients
        self.__optimizer.zero_grad()
        # Feed-forward pass
        features = self.__model.get_feature_extraction_output(inputs)
        outputs = self.__model.get_classification_output(features)

        if self.__hyperparameters.get_num_classes() > 1:
          # torch.nn.CrossEntropyLoss() requires the type of y_true is torch.long
          labels = labels.long()

        labels = labels.squeeze() # Remove any additional dimensionality from the data
        if self.__hyperparameters.get_num_classes() > 1:
          loss = self.__criterion(outputs, labels)
        else:
           loss = self.__criterion(outputs, labels.unsqueeze(1).float())

        # Backpropagation
        loss.backward()
        self.__optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        self.__metric_tracker._update(outputs, labels, loss)

        if self.__hyperparameters.get_save_features():
          if epoch == self.__hyperparameters.get_num_epochs() - 1:
            self.__extracted_features.append(features.detach().cpu().numpy())
            self.__extracted_labels.append(labels.detach().cpu().numpy())

      # Compute evaluation metrics at the end of the i-th training epoch
      epoch_loss = running_loss / len(train_loader.dataset)
      precision, recall, f1, auc_roc, precision_recall_curve = self.__metric_tracker._compute()
      print(f"Epoch {epoch}/{self.__hyperparameters.get_num_epochs() - 1}, Loss: {epoch_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUROC: {auc_roc:.4f}")
      
      # Store values for i-th epoch. These values will be plotted at the end of training phase
      training_epoch.append(epoch)
      training_epoch_loss.append(epoch_loss)
      training_precision.append(precision)
      training_recall.append(recall)
      training_f1.append(f1)
      training_auc_roc.append(auc_roc)
    # end training
    self.__metric_tracker._plot_training_metrics(training_epoch, training_epoch_loss, training_precision, training_recall, training_f1, training_auc_roc, self.__hyperparameters._plot_dir)

    save_path = self.__hyperparameters._local_path+'trained_models/cnn'+self.__hyperparameters._training_set_name+'.pt'
    torch.save(self.__model.state_dict(), save_path)

    if self.__hyperparameters.get_save_features():
      if self.__extracted_features:
        features_array = np.vstack(self.__extracted_features)
        labels_array = np.concatenate(self.__extracted_labels)
        np.save(self.__features_filename+'_extracted_features.npy', features_array)
        np.save(self.__features_filename+'_extracted_labels.npy', labels_array)
        try:
            print(f"features shape: {features_array.shape}")
            print(f"labels shape: {labels_array.shape}")
        except Exception:
            print(f"Error while reading features shape\n")
        print(f"Extracted features stored in: {self.__features_filename}")
        print(f"Labels stored in: {self.__features_filename}")

    return self.__model


  def evaluate(self, dataset_path, model_path=''):
    """
      Compute the main evaluation metrics for the given model: loss, precision, recall, f1score, area under roc and confusion matrix. 
      For binary classification problems, precision-recall and ROC curves are computed.
    """
    print(f'Model assessment on test set:{self.__hyperparameters.get_test_set_path()}')

    # If model_path is not given, then you already have the Model object that had been trained
    if model_path:
      print(f'Loading the model stored in:{model_path}')
      self.__model.load_state_dict(torch.load(model_path))

    if self.__hyperparameters.get_num_classes() == 1:
      # In test mode for binary classification, set the loss function without class weighting
      self.__criterion = nn.BCEWithLogitsLoss()

    # Load test data
    test_loader, y_test = self.__load_dataset(dataset_path,'test')

    # Set model into evaluation mode
    self.__model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    # Disable gradients computation during model assessment
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Get (x_test, y_test) data
            inputs, labels = inputs.to(device), labels.to(device)
            # Feed-forward pass
            features = self.__model.get_feature_extraction_output(inputs)
            outputs = self.__model.get_classification_output(features)

            # Compute loss
            if self.__hyperparameters.get_num_classes() > 1:
              loss = self.__criterion(outputs, labels)
            else:
              loss = self.__criterion(outputs, labels.unsqueeze(1).float())  # Use this for single neuron output
            running_loss += loss.item() * inputs.size(0)

            # i. Change loss function based on the number of output classes: Binary --> sigmoid; Multi-class --> softmax.
            # ii. Manage the y_pred values (i.e. stored into the list all_preds) in base alla tipologia di classificazione.
            # ii. Transfer y_pred and y_true from GPU to CPU for evaluation metrics computation
            # iii. Finally, concatenate all model's predictions and labels from the batch of data stored in all_preds[] and all_labels[]
            if self.__hyperparameters.get_num_classes() > 1:
              probs = torch.softmax(outputs, dim=1)         # i. softmax
              preds = torch.argmax(probs, dim=1)            
              all_preds.append(preds.cpu().numpy())         # ii.
              all_probs.append(probs.cpu().numpy())
            else:
              preds = torch.round(torch.sigmoid(outputs))   # i. sigmoid. Round model's predictions to 0 or 1
              all_preds.append(preds.cpu().numpy())         # ii.
            
            all_labels.append(labels.cpu().numpy())         # ii. same operation, no matter the number of output classes

    if self.__hyperparameters.get_num_classes() > 1:        # iii.
      all_preds = np.concatenate(all_preds)                 
      all_probs = np.concatenate(all_probs, axis=0)         # iii. For multi-class classification, the computation of roc_auc_score requires 2D array of probability scores the model computed for each class 
    else:
      all_preds = np.concatenate(all_preds)
    
    all_labels = np.concatenate(all_labels)

    # Compute the average loss over the entire dataset
    test_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    self.__metric_tracker.compute_evaluation_metrics(all_preds, all_probs, all_labels)

    if self.__hyperparameters.get_num_classes() == 1:
      #self.__metric_tracker.plot_prcurve(all_preds, all_labels, self.__hyperparameters._plot_dir)
      self.__metric_tracker.plot_roc(all_preds, all_labels, self.__hyperparameters._plot_dir)



def main_dartvetter():
    """
        # Important information about input variables
        ### features_filename filename structure: <training_set_name>_extracted_features.npy or <training_set_name>_extracted_labels.npy
        ### here you just have to add the prefix <training_set_name> because the suffix is automatically added within the train method in
        ### the train_test_cnn_class.py file
    """
    # Get hyperparameters
    hyperparameters_object = InputVariables.get_input_hyperparameters('config_cnn.yaml')
    # Print hyperparameters
    for field, value in hyperparameters_object.__dict__.items():
        print(f"{field}: {value}")
    # Create the model architecture
    dartvetter = m1_class.DartVetterFiscale2024(
        hyperparameters_object.get_cbn(),
        hyperparameters_object.get_cfs(),
        hyperparameters_object.get_cbff(),
        hyperparameters_object.get_psz(),
        hyperparameters_object.get_pst(),
        hyperparameters_object.get_rho(),
        hyperparameters_object.get_fc_layers_num(),
        hyperparameters_object.get_fc_units(),
        hyperparameters_object.get_fc_output_size()
        )
    # Train the model by using the method of the class Model
    model = Model(dartvetter, hyperparameters_object)
    
    model.train(hyperparameters_object.get_training_set_path())
    print("Training completed.")
    
    model.evaluate(hyperparameters_object.get_test_set_path())

    exit(0)


if __name__ == '__main__':
    main_dartvetter()
