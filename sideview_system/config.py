dlc_training_datasets = {
'channel_3': './dlc_cow_models/cow_v3rgb-20191008/training-datasets/iteration-0/UnaugmentedDataSet_cow_channel_3_v3Oct8/CollectedData_He.csv',
'merge': './dlc_cow_models/cow_3_data_v3-He-2019-10-08/training-datasets/iteration-0/UnaugmentedDataSet_cow_3_data_v2Oct8/CollectedData_He.csv',
'merge_underbelly': "./dlc_cow_models/cow_3_data_v4-Underbelly-Shaan-2023-11-24/training-datasets/iteration-0/UnaugmentedDataSet_cow_3_data_v2Oct8/CollectedData_He.csv"
}

# channel 3 videos
channel_3_model_color = {'folder':'../dlc_cow_models/cow_v3rgb-20191008/',
				'dlc_cfg':'dlc-models/iteration-0/cow_channel_3_v3Oct8-trainset95shuffle1/test/pose_cfg.yaml',
				'init_weights': 'dlc-models/iteration-0/cow_channel_3_v3Oct8-trainset95shuffle1/train/snapshot-60000'}
channel_3_model_dif = {'folder':'../dlc_cow_models/cow_v3diff-20191008/',
			'dlc_cfg':'dlc-models/iteration-0/cow_channel_3_v3_difOct8-trainset95shuffle1/test/pose_cfg.yaml',
			'init_weights': 'dlc-models/iteration-0/cow_channel_3_v3_difOct8-trainset95shuffle1/train/snapshot-45000'}
channel_3_trained_models = {'color_model': channel_3_model_color, 'dif_model': channel_3_model_dif}

# merged train
merged_model_color = {'folder':'./dlc_cow_models/cow_3_data_v3-He-2019-10-08/',
 				'dlc_cfg':'dlc-models/iteration-0/cow_3_data_v2Oct8-trainset95shuffle1/test/pose_cfg.yaml',
 				'init_weights': 'dlc-models/iteration-0/cow_3_data_v2Oct8-trainset95shuffle1/train/snapshot-30000'}
merged_model_dif = {'folder':'./dlc_cow_models/cow_3_data_dif_v3-He-2019-10-08/',
 			'dlc_cfg':'dlc-models/iteration-0/cow_3_data_dif_v2Oct8-trainset95shuffle1/test/pose_cfg.yaml',
 			'init_weights': 'dlc-models/iteration-0/cow_3_data_dif_v2Oct8-trainset95shuffle1/train/snapshot-30000'}
merged_trained_models = {'color_model': merged_model_color, 'dif_model': merged_model_dif}

merged_underbelly_model_color = {'folder':'./dlc_cow_models/cow_3_data_v4-Underbelly-Shaan-2023-11-24/',
 				'dlc_cfg':'dlc-models/iteration-0/cow_3_data_v2Oct8-trainset95shuffle1/train/pose_cfg.yaml',
 				'init_weights': 'dlc-models/iteration-0/cow_3_data_v2Oct8-trainset95shuffle1/train/snapshot-45000'}
merged_underbelly_model_dif = {'folder':'./dlc_cow_models/cow_3_data_dif_v4-Underbelly-Shaan-2023-11-24/',
 			'dlc_cfg':'dlc-models/iteration-0/cow_3_data_dif_v2Oct8-trainset95shuffle1/test/pose_cfg.yaml',
 			'init_weights': 'dlc-models/iteration-0/cow_3_data_dif_v2Oct8-trainset95shuffle1/train/snapshot-45000'}
merged_underbelly_trained_models = {'color_model': merged_underbelly_model_color, 'dif_model': merged_underbelly_model_dif}

# Access the trained DLC models through this structure
trained_models_dict = {'channel_3': channel_3_trained_models,
					   'merge': merged_trained_models,
					   'merge_underbelly': merged_underbelly_trained_models}

class Config:
	def __init__(self, model_set):
		self.model_set = model_set
		self.cow_centroid_model_data = dlc_training_datasets[model_set]
		self.cow_centroid_model = None
		self.body_region_points = None
		self.head_region_points = None
		self.leg_region_points = None
		self.body_limbs = None
		self.leg_limbs = None
		self.initialize_settings()
	
	def initialize_settings(self):
		if self.model_set in ['channel_3', 'merge']:
			
			self.body_region_points = ['nose', 'head', 'necktop', 'shoulder', 'spine', 'tailbase', 
                                       'bottomback', 'bottomfront', 'neckbot']
									   
			self.head_region_points = ['nose', 'head', 'necktop', 'neckbot']
			
			self.leg_region_points = ["leftfrontleg", "leftfronthoof", "rightfrontleg", "rightfronthoof", 
                                      "leftbackleg", "leftbackhoof", "rightbackleg", "rightbackhoof"]
			self.body_limbs = [['nose', 'head'],
                               ['head', 'necktop'],
                               ['necktop', 'shoulder'],
                               ['shoulder', 'spine'],
                               ['spine', 'tailbase'],
                               ['tailbase', 'bottomback'],
                               ['bottomback', 'bottomfront'],
                               ['bottomfront', 'neckbot'],
                               ['neckbot', 'nose']]
			self.leg_limbs = [['shoulder', 'leftfrontleg'],
                              ['leftfrontleg', 'leftfronthoof'],
                              ['shoulder', 'rightfrontleg'],
                              ['rightfrontleg', 'rightfronthoof'],
                              ['tailbase', 'leftbackleg'],
                              ['leftbackleg', 'leftbackhoof'],
                              ['tailbase', 'rightbackleg'],
                              ['rightbackleg', 'rightbackhoof']]
		
		# Additional settings for 'merge_underbelly'
		if self.model_set == 'merge_underbelly':
			
			self.body_region_points = ['nose', 'head', 'necktop', 'shoulder', 'spine', 'tailbase', 
                                       'bottomback', 'bottomfront', 'neckbot', 'underbelly']
			self.body_limbs = [['nose', 'head'],
                               ['head', 'necktop'],
                               ['necktop', 'shoulder'],
                               ['shoulder', 'spine'],
                               ['spine', 'tailbase'],
                               ['tailbase', 'bottomback'],
                               ['bottomback', 'underbelly'],
                               ['underbelly', 'bottomfront'],
                               ['bottomfront', 'neckbot'],
                               ['neckbot', 'nose']]
			self.head_region_points = ['nose', 'head', 'necktop', 'neckbot']
			
			self.leg_region_points = ["leftfrontleg", "leftfronthoof", "rightfrontleg", "rightfronthoof", 
                                      "leftbackleg", "leftbackhoof", "rightbackleg", "rightbackhoof"]

			self.leg_limbs = [['shoulder', 'leftfrontleg'],
                              ['leftfrontleg', 'leftfronthoof'],
                              ['shoulder', 'rightfrontleg'],
                              ['rightfrontleg', 'rightfronthoof'],
                              ['tailbase', 'leftbackleg'],
                              ['leftbackleg', 'leftbackhoof'],
                              ['tailbase', 'rightbackleg'],
                              ['rightbackleg', 'rightbackhoof']]

		if self.cow_centroid_model is None:
			from .cow_centroid_model import generate_cow_centroid_model
			self.cow_centroid_model = generate_cow_centroid_model(self)

