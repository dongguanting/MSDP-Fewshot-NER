CUDA_VISIBLE_DEVICES=0 python aa.py \
--do_train \
--batch_size 8  \
--train_data /aa/bb/cc/dd/inter_train_5_1_seq_in.txt \
--train_data2 /aa/bb/cc/dd/inter_train_5_1_seq_out.txt \
--model_save_dir ./ee/ \
--epoch 5  \
