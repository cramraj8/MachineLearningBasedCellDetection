

DataPrep
    - has scripts to read the data(JPEG & XML) and prepare them towards
    train:test split.

Evaluation
    - This has scripts to run evaluation on sample(1 sample) JSON & XML files.
    - Also it has some other reference files(images, predicted images)

CompleteData
    |
    |-- InputData
    |       This has all JPEG of ROIs and XML of Annotations.
    |
    |-- Outputs
        |
        |---- TEST_OUTPUT1_th0.0 =>
        |        This has basic predicted data
        |        main output file used for evaluation.
        |        Prediction outputs at threshold(0.0) - (no thresholding applied)
        |
        |---- TEST_OUTPUT1_th0.1 =>
                anoter sample output folder thresholded at 0.1


If need to get total Precision & Recall estimation, we need to include all
17 ROIs' output JSON files with their respective XMLs.
You can get JSONs from 'CompleteData/Outputs/TEST_OUTPUT1_th0.0/'
You can get XMLs from 'DataPrep/XML_Objectness/'
You have to 'S3_create_XMLObjectness.py' run script to create the 'XML_Objectness' folder.
