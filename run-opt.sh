# ./mainV5.py --tbs=8 --ag_step=1 --lr0=1e-4 --seed=0 --optimizer=radam --dtoSGD=0 --amsgrad=0 --norm_layer=in > lr1-bs8-radam-inorm.log
# ./mainV5.py --tbs=8 --ag_step=1 --lr0=1e-4 --seed=0 --optimizer=radam --dtoSGD=0 --amsgrad=1 --norm_layer=in > lr1-bs8-radam-amsg-inorm.log
# ./mainV5.py --tbs=8 --ag_step=1 --lr0=1e-4 --seed=0 --optimizer=radam --dtoSGD=1 --amsgrad=0 --norm_layer=in > lr1-bs8-radam_sgd-inorm.log
# ./mainV5.py --tbs=8 --ag_step=1 --lr0=1e-4 --seed=0 --optimizer=radam --dtoSGD=1 --amsgrad=1 --norm_layer=in > lr1-bs8-radam_sgd-amsg-inorm.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-4 --mom=0.9 --nag=1 --seed=0 --norm_layer=in > lr4-m9-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.9 --nag=1 --seed=0 --norm_layer=in > lr3-m9-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=in > lr2-m9-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-1 --mom=0.9 --nag=1 --seed=0 --norm_layer=in > lr1-m9-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-4 --mom=0.99 --nag=1 --seed=0 --norm_layer=in > lr4-m99-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=0 --norm_layer=in > lr3-m99-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.99 --nag=1 --seed=0 --norm_layer=in > lr2-m99-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-1 --mom=0.99 --nag=1 --seed=0 --norm_layer=in > lr1-m99-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-4 --mom=0.999 --nag=1 --seed=0 --norm_layer=in > lr4-m999-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.999 --nag=1 --seed=0 --norm_layer=in > lr3-m999-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.999 --nag=1 --seed=0 --norm_layer=in > lr2-m999-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-1 --mom=0.999 --nag=1 --seed=0 --norm_layer=in > lr1-m999-bs8-sgd-nag.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=1 --norm_layer=in > lr3-m99-bs8-sgd-nag1.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=2 --norm_layer=in > lr3-m99-bs8-sgd-nag2.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=3 --norm_layer=in > lr3-m99-bs8-sgd-nag3.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=4 --norm_layer=in > lr3-m99-bs8-sgd-nag4.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-3 --mom=0.99 --nag=1 --seed=5 --norm_layer=in > lr3-m99-bs8-sgd-nag5.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=in > lr2-m9-bs8-sgd-nag1.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=in > lr2-m9-bs8-sgd-nag2.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=in > lr2-m9-bs8-sgd-nag3.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=in > lr2-m9-bs8-sgd-nag4.log
# ./mainV5.py --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=in > lr2-m9-bs8-sgd-nag5.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn0.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn1.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn2.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn3.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn4.log
# ./mainV5-sn.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-snorm-nobn5.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws0.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws1.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws2.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws3.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws4.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-inorm-ws5.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws0.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws1.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws2.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws3.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws4.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-snorm-ws5.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn0.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn1.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn2.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn3.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn4.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=frn --leps=0 > lr2-m9-bs8-sgd-nag-frn5.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps0.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=1 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps1.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=2 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps2.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=3 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps3.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=4 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps4.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=5 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps5.log
# ./mainV5_norm.py --train_f=train.lst --val_f=val.lst --test_f=test.lst --tbs=8 --ag_step=1 --optimizer=sgd --lr0=1e-2 --mom=0.9 --nag=1 --seed=0 --norm_layer=frn --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps.log
# ./mainV5_norm_ws.py --seed=0 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=gn --num_groups=16 > lr2-m9-bs8-sgd-nag-gn-ng16-5.log
# ./mainV5_norm_ws.py --seed=0 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=gn --group_size=2 > lr2-m9-bs8-sgd-nag-gn-gs2-5.log
# ./mainV5_norm_ws.py --seed=0 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=in --wstd=1 > lr2-m9-bs8-sgd-nag-in-ws-5.log
# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 > lr2-m9-bs8-sgd-nag-sn-ws-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --using_movavg=0 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --using_movavg=1 --using_bn=0 > lr2-m9-bs8-sgd-nag-sn-ma-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --using_movavg=0 --using_bn=1 > lr2-m9-bs8-sgd-nag-sn-bn-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=in --in_with_mom=1 > lr2-m9-bs8-sgd-nag-in-mom-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=in > lr2-m9-bs8-sgd-nag-in-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=frn --act_layer=tlu --eps=1e-6 --leps=0 > lr2-m9-bs8-sgd-nag-frn-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=frn --act_layer=tlu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=frn --act_layer=relu --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-relu-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=frn --act_layer=swish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-swish-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=frn --act_layer=mish --eps=1e-4 --leps=1 > lr2-m9-bs8-sgd-nag-frn-leps-mish-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --act_layer=swish > lr2-m9-bs8-sgd-nag-sn-ws-swish-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --act_layer=mish > lr2-m9-bs8-sgd-nag-sn-ws-mish-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=16 > lr2-m9-bs8-16-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=32 > lr2-m9-bs8-32-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=64 > lr2-m9-bs8-64-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=128 > lr2-m9-bs8-128-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=256 > lr2-m9-bs8-256-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=512 > lr2-m9-bs8-512-sgd-nag-mabn-5.log

# ./mainV5_mabn.py --seed=0 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-0.log
# ./mainV5_mabn.py --seed=1 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-1.log
# ./mainV5_mabn.py --seed=2 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-2.log
# ./mainV5_mabn.py --seed=3 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-3.log
# ./mainV5_mabn.py --seed=4 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-4.log
# ./mainV5_mabn.py --seed=5 --norm_layer=mabn --tbs_target=1024 > lr2-m9-bs8-1024-sgd-nag-mabn-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=bn > lr2-m9-bs8-sgd-nag-bn-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=in --affine=0 > lr2-m9-bs8-sgd-nag-in-noaff-0.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-ws-gc-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --gc=1 --gcc=1 > lr2-m9-bs8-sgd-nag-sn-gc-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-1 > lr2-m9-bs8-sgd-nag-sn-ws-wd1-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-2 > lr2-m9-bs8-sgd-nag-sn-ws-wd2-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-4 > lr2-m9-bs8-sgd-nag-sn-ws-wd4-5.log

# ./mainV5_norm_ws.py --seed=0 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-0.log
# ./mainV5_norm_ws.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-1.log
# ./mainV5_norm_ws.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-2.log
# ./mainV5_norm_ws.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-3.log
# ./mainV5_norm_ws.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-4.log
# ./mainV5_norm_ws.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-5 > lr2-m9-bs8-sgd-nag-sn-ws-wd5-5.log

# ./mainV5_norm_ws_xa.py --seed=0 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa0.log
# ./mainV5_norm_ws_xa.py --seed=1 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa1.log
# ./mainV5_norm_ws_xa.py --seed=2 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa2.log
# ./mainV5_norm_ws_xa.py --seed=3 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa3.log
# ./mainV5_norm_ws_xa.py --seed=4 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa4.log
# ./mainV5_norm_ws_xa.py --seed=5 --norm_layer=sn --wstd=1 --n_channels=4 --n_filters=32 --n_class=7 --tbs=16 --vbs=64 --data_path=/home/lz/data/AIDataset/ --model_path=model/AIDataset/ --train_f=train_new.lst --val_f=val_new.lst > xa5.log

# ./mainV5_norm_ws_lrd.py --norm_layer=sn --wstd=1 --wd=1e-3 --max_epochs=200 --anneal_start=120 > run3.log
#./mainV5_norm_ws_lrd.py --seed=1 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-lrd1.log
#./mainV5_norm_ws_lrd.py --seed=2 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-lrd2.log
#./mainV5_norm_ws_lrd.py --seed=3 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-lrd3.log
#./mainV5_norm_ws_lrd.py --seed=4 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-lrd4.log
#./mainV5_norm_ws_lrd.py --seed=5 --norm_layer=sn --wstd=1 --wd=1e-3 > lr2-m9-bs8-sgd-nag-sn-ws-wd3-lrd5.log

./mainV5_norm_ws_lrd.py --norm_layer=sn --wstd=1 --wd=1e-3 --max_epochs=60 --anneal_start=20 --data_path=/data/20201018/ --model_path=/data/20201018/model/ --save_epoch=10 > run4.log
