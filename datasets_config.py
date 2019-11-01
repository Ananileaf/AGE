datasets_config = {
						'dataset_path':'/home/zuowenhang/datasets/AGE',#相对路径可能报错
						'img_size':998,
    					'rotate_angle':15, #[-rotate_angle,rotate_angle]
    					'num_workers':2, #
    					'batch_size':4, #
    					'SNR':0.95, #椒盐噪声,[0.0,1.0] 1不触发
    					'GaussianBlur_sigma':1.0, #高斯噪声，0不触发
    					'Noisy_prob':0.5, #[0,0.5)触发椒盐，[0.5,1]触发高斯
}
