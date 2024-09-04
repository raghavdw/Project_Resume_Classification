import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline


from resume_model.config.core import config
from resume_model.processing.features import labelEncoder


resume_pipe=Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.model_config.embarked_var)
     ),
     ##==========Mapper======##
     ("map_sex", Mapper(config.model_config.gender_var, config.model_config.gender_mappings)
      ),
     ("map_embarked", Mapper(config.model_config.embarked_var, config.model_config.embarked_mappings )
     ),
     ("map_title", Mapper(config.model_config.title_var, config.model_config.title_mappings)
     ),
     # Transformation of age column
     ("age_transform", age_col_tfr(config.model_config.age_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
     ])