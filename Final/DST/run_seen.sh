python3 test_unseen.py --test_file "${1}" --mode "test_seen"
# dst.py will read data(after predict service) from './data_after_pred_serv'
python3 dst.py do_training=false model.dataset.task_name=seen_domain
python3 format.py --mode "test_seen" --result_file "${2}"