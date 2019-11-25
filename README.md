# MachineLearningBasedCellDetection
Machine Learning based cell detection and classification framework implemented towards the publication.


1. Detection
    1.1 DataPrep

        set SEED = 27
        set ROUND = 1
        OverallReports/S4_arrange_DATAFolders.py

    1.2 luminoth-master

        For each round, move jobs into jobs1/jobs2/...

        - $ lumi transform
                tf_dataset1
                change
                    sample_config.yml
                    base_config.yml


                    lumi dataset transform \
                        --type pascal \
                        --data-dir ../DataPrep/DATAFolder5/ \
                        --output-dir tf_dataset5/ \
                        --split train

                    objectness: 6087
                    tf_dataset1/train.tfrecords"

        - $ lumi train

                    lumi train -c examples/sample_config.yml

        - $ lumi predict

                    lumi predict -c examples/sample_config.yml path-or-dir ../DataPrep/DATAFolder4/TEST_DATAFolder/JPEG --save-media-to ../OutputPrediction/TEST_OUTPUT4/JPEG/ --min-prob 0.0 --max-detections 500 --output ../OutputPrediction/TEST_OUTPUT4/JSON/

    1.3 Evaluation

        DiscreteMeasures.py

        Outputs:    TP_match_table.csv
                    Prob_table.csv

    1.4 Results

        plot_PRCurve.py
        save PR Curve.png
        save AUC value in a file.


2. Classification
    2.1 params.py
            ROUND = 1
            SEED(FLAGS) = 27

            change,
            n_train = 7353
            n_test = 1522

    2.2 create_TF_records.py

    2.3 inputs.py

    2.4 main_VGG16_withoutsummary.py

    2.5 cluster_prediction.py

    2.6 plotMultiROC_local.py

    2.7 evalCompactClasses.py


3. Linker
    3.1 CropCells.py

    <!-- 3.2 Classification/testCropCells.py -->


TODO:
1. Documentation - Readme
2. Writing DocString for each script
