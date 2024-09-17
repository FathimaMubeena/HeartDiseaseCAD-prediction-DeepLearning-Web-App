# Predicting Coronary Artery Disease using a Keras neural network
## Project Purpose:
Heart diseases are often underestimated, but, in reality, they are the leading cause of death in the world. Among them, coronary artery disease (CAD) accounts for about a third of all deaths worldwide in people over 35 years of age. CAD is the result of arteriosclerosis, which consists in the narrowing of the blood vessels and the hardening of its walls. In some cases, CAD can completely block the influx of oxygen-rich blood to the heart muscle, causing a heart attack.

**Dataset:** Heart Disease Data Set is extracted from the  UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/45/heart+disease

This dataset is provided by the following four clinical institutions:
1. Cleveland Clinic Foundation (CCF),
2. Hungarian Institute of Cardiology (HIC),
3. Long Beach Medical Center (LBMC), and
4. University Hospital in Switzerland (SUH).

The dataset contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by the Machine Learning community to this date.

The `goal` field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
1. 0: no presence
2. 1,2,3,4: presence

### Business Case
The create a ML model that acurately predict a condition of heart disease in a patient. This model will be used by doctors to help them in the diagnosis of heart disease.

### Goal:
To accurately predict the presence of heart disease in a patient using a Keras neural network without falsely diagnosing a patient with heart disease when they do not have it. (i.e. minimizing false positives)

### Deliverables:
1. A Keras neural network model that predicts the presence of heart disease in a patient.
2. A web application that allows doctors to input patient data and get a prediction of the presence of heart disease in the patient.

# How to Run the Project
1. Clone the repository:
   ```shell
   git clone
   ```
2. Create a virtual environment:
   ```shell
    python3 -m venv heart_disease_prediction_env
    ```
2. Activate the virtual environment:  
   On macOS:
      ```shell
      source heart_disease_prediction_env/bin/activate
      ```
3. Install the required packages:
   ```shell
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
    ```shell
    streamlit run src/main/deploy/deploy_streamlit.py
    ```

## Results and Model Evaluation:

Deployed model :
![result_Streamlit_CNN_prediction.png](src%2Fmain%2Fresources%2Fresult_Streamlit_CNN_prediction.png)


## Business Impact:
End users will be able to use the web app built off of this model to predict the presence of coronary artery disease (CAD) in patients. This will enable doctors to make more informed decisions quickly, improving patient outcomes. The model's high accuracy ensures that true cases of CAD are identified, while minimizing false positives, thus enhancing the reliability of diagnoses. This tool will streamline the diagnostic process, allowing healthcare providers to serve more patients efficiently, ultimately leading to better healthcare delivery and increased patient satisfaction

### Next Steps: