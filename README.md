# Semi-Supervised-Clustering-using-Neural-Networks

## Instructions to Run

**1. Clone the repository**
      
      $ git clone https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks.git
      $ cd Semi-Supervised-Clustering-using-Neural-Networks
      
**2. Create new virtual environment**

      $ sudo pip install virtualenv
      $ virtualenv venv
      $ source venv/bin/activate
      $ pip install -r requirements.txt
      
**3. Repository Information**

<ul>
    <li> The file <b> Test.py </b> contains the code for testing the auto-encoder. </li>
    <li> The file <b> DataLoader.py </b> contains the different data loader functions. </li>
    <li> The file <b> PreTraining.py </b> contains the code for pretraining the auto-encoder.</li>
    <li> The file <b> Visualisations.py </b> contains the code for producing the plots present in the report. </li>
    <li> The file <b> Train.py </b> contains the code for training the auto-encoder on the custom loss function. </li>
    <li> The file <b> Model.py </b> contains the architecture definition of the convolutional auto-encoder that is used.       </li>
</ul>

**4. Training the auto-encoder network**
      
      $ python Train.py --dataSet <dataSet Name> --percentLabData <Percentage of labelled data to be used>
      
The training procedure will train the convolutional auto-encoder on the specified dataset. The folder <b> /TrainingResults </b> will get populated with the trained model as well as various other information required for generating the plots.
  
**5. Testing the auto-encoder network**

      $ python Test.py --dataSet <dataSet Name> --percentLabData <Percentage of labelled data to be used>

The testing script will run the trained model on the test data and will populate the folder <b> /TestingResults </b>.

**5. Producing the plots**

      $ python Visualisations.py --dataSet <dataSet Name> --percentLabData <Percentage of labelled data to be used>

This script will produce the different plots for the dataset and percentage of labelled data specified in the input.

**6. Example Plots**

The plots generated for 10% labelled data for MNIST are displayed below.

<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/AnnealingLoss.png">
<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/DifferentLoss1.png">
<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/DifferentLoss2.png">
<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/NMIPurity.png">
<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/NMIPurityAnnealing.png">
<img src="https://github.com/RahulBaboota/Semi-Supervised-Clustering-using-Neural-Networks/blob/master/Images/totalLoss.png">



