python get_jmjx_tiku_features.py --part=0 &
python get_jmjx_tiku_features.py --part=1 &
python get_jmjx_tiku_features.py --part=2 &
python get_jmjx_tiku_features.py --part=3 &
python get_jmjx_tiku_features.py --part=4 &
python get_jmjx_tiku_features.py --part=5 &
python get_jmjx_tiku_features.py --part=6 &
python get_jmjx_tiku_features.py --part=7 &

wait

python eval_local_img_milvus_zcl.py