# End-to-end-Chest-Cancer-Classification-

## Workflow
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

### dvc dag

+----------------+            +--------------------+ 
| data_ingestion |            | prepare_base_model | 
+----------------+*****       +--------------------+ 
         *             *****             *
         *                  ******       *
         *                        ***    *
         **                        +----------+      
           **                      | training |      
             ***                   +----------+      
                ***             ***
                   **         **
                     **     **
                  +------------+
                  | evaluation |
                  +------------+



### DagsHub For Tracking Experiments : https://dagshub.com/omarhatem44/End-to-end-Chest-Cancer-Classification.mlflow/#/experiments/3/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D