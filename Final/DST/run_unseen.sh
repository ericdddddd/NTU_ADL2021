python3 test_unseen.py --test_file "${1}" --mode "test_unseen"
python3 dst.py do_training=false model.dataset.task_name=unseen_domain
python3 format.py --mode "test_unseen" --result_file "${2}"