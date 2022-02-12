WORKLOAD=$1
TUNING_METHOD=$2
KNOB_NUM=$3
SELECTION_METHODS=('shap' 'fanova' 'lasso' 'ablation' 'rf' )
ALL_KNOBS="experiment/gen_knobs/mysql_all_197.json"
SURROGATE_DATA="../tuning_benchmark/data/${WORKLOAD}_197knob.csv"

if [ "$WORKLOAD" = "JOB" ] ; then
  Y="lat"
  echo $Y
else
  Y="tps"
fi

for method in ${SELECTION_METHODS[@]}
do
  #CMD="experiment/exp_importance_${method}.py"
  OUTPUT_KNOBS="experiment/gen_knobs/${WORKLOAD}_${method}.json"
  SURROGATE_MODEL_DIR="../tuning_benchmark/surrogate/${WORKLOAD}_all.joblib"
  LHS_LOG="result/${WORKLOAD}_${KNOB_NUM}knobs_${TUNING_METHOD}_${method}.result"
  #python $CMD $ALL_KNOBS  $SURROGATE_DATA $OUTPUT_KNOBS
  #python experiment/surrogate_model_fit.py $OUTPUT_KNOBS  $SURROGATE_DATA $SURROGATE_MODEL_DIR $KNOB_NUM

  FILE="result/"
  IFS='/' read -ra ADDR <<< "$LHS_LOG"
  SUB="${ADDR[-1]}"
  ID_MAX=-1

  for f in $(ls $FILE); do
    if [[ "$f" == *"$SUB"* ]]; then
      IFS='.' read -ra ADDR <<< "$f"   # str is read into an array as tokens separated by IFS
      ID="${ADDR[-1]}"
      if [ $ID -gt $ID_MAX ]; then
        ID_MAX=$ID
      fi
    fi
  done

  ID_MAX=$(($ID_MAX + 1))
  echo $ID_MAX
  LHS_LOG_MORE="$LHS_LOG.$ID_MAX"

  python run_benchmark.py --method=$TUNING_METHOD  --knobs_config=$OUTPUT_KNOBS  --model_path=$SURROGATE_MODEL_DIR --knobs_num=$KNOB_NUM   --lhs_log=$LHS_LOG_MORE --y_variable=$Y --tr_init
  python experiment/result_parse.py  $LHS_LOG_MORE $Y
done

