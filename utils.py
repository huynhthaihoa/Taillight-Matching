from enum import Enum

'''List of bus taillights in left images'''
bus_map_left = {'1.JPG': ['243'], '2.JPG': ['198'], '3.JPG': ['204'], '4.JPG': ['110'], '5.JPG': ['166'], '6.JPG': ['160'], 
'7.JPG': ['159'], '8.JPG': ['139'], '9.JPG': ['153'], '10.JPG': ['129', '144'], '11.JPG': ['164', '181'], '12.JPG': ['131', '141'],
'13.JPG': ['158', '166'], '14.JPG': ['151', '176'], '15.JPG': ['164', '178', '190'], '16.JPG': ['139', '152', '167'], '17.JPG': ['154', '169', '183'],
'18.JPG': ['140', '166', '180'], '19.JPG': ['130'], '20.JPG': ['184', '203', '208', '214'], '21.JPG': ['160', '182', '180', '187'], 
'22.JPG': ['153', '181', '184', '190'], '23.JPG': ['162', '181', '188', '192'], '24.JPG': ['154', '159', '165'], 
'25.JPG': ['135', '151', '158', '164'], '26.JPG': ['158', '168', '177', '182'], '27.JPG': ['160', '167', '176']}

'''List of bus taillights in right images'''
bus_map_right = {'1.JPG': ['90'], '2.JPG': ['47'], '3.JPG': ['72'], '4.JPG': ['42'], '5.JPG': ['116'], '6.JPG': ['93'], 
'7.JPG': ['154'], '8.JPG': ['54'], '9.JPG': ['131'], '10.JPG': ['93', '101'], '11.JPG': ['66', '71'], '12.JPG': ['100', '105'],
'13.JPG': ['50', '53'], '14.JPG': ['31', '35'], '15.JPG': ['62', '64', '66'], '16.JPG': ['55', '57', '60'], '17.JPG': ['64', '66', '67'],
'18.JPG': ['34', '35', '36'], '19.JPG': ['35'], '20.JPG': ['34', '36', '38', '40'], '21.JPG': ['39', '40', '41', '42'], 
'22.JPG': ['35', '37', '38', '39'], '23.JPG': ['41', '44', '46', '48'], '24.JPG': ['44', '45', '46'], 
'25.JPG': ['55', '57', '58', '60'], '26.JPG': ['36', '38', '40', '42'], '27.JPG': ['53', '55', '57']}

'''List of taillights for stereo matching in left images'''
stereo_matching_left = {'1.JPG': ['243', '262', '263', '251', '219', '231'], '2.JPG': ['198', '219', '220', '182'], 
'3.JPG': ['204', '237', '224', '225', '214'], '4.JPG': ['140', '142', '110', '131', '132', '121'], 
'5.JPG': ['166', '213', '219', '228', '199', '200', '157'], '6.JPG': ['160', '220', '228', '249', '211', '213', '165'], 
'7.JPG': ['159', '142', '192', '204', '227', '185'], '8.JPG': ['139', '126', '182', '192', '213', '174'], 
'9.JPG': ['153', '184', '197', '241'], '10.JPG': ['129', '144', '155', '160', '191', '194', '230'], 
'11.JPG': ['164', '181', '189', '235', '237'], '12.JPG': ['131', '141', '145', '166', '177'],
'13.JPG': ['158', '166', '203', '208', '218', '222', '228'], '14.JPG': ['151', '176', '214', '226', '230', '234', '253'], 
'15.JPG': ['164', '178', '190', '230', '233', '237', '241', '255'], '16.JPG': ['139', '152', '167', '222', '232', '238', '255'], 
'17.JPG': ['154', '169', '183', '224', '226', '229', '235', '250'], '18.JPG': ['140', '166', '180', '221', '222', '225', '229', '238'], 
'19.JPG': ['130', '215', '221', '225', '240'], '20.JPG': ['184', '203', '208', '214', '256', '260', '274'], 
'21.JPG': ['160', '182', '180', '187', '237', '241', '245', '256'], '22.JPG': ['153', '181', '184', '190', '238', '240', '243', '246', '258'], 
'23.JPG': ['162', '188', '181', '192', '232', '237', '244', '252'], '24.JPG': ['154', '159', '165', '205', '211', '228'], 
'25.JPG': ['135', '151', '158', '164', '201', '205', '212', '220'], '26.JPG': ['158', '168', '177', '182', '215', '216', '219', '223', '235'], 
'27.JPG': ['160', '167', '176', '227', '228', '232', '237', '246'], '29.JPG': ['241', '215'], '30.JPG': ['184', '186'], 
'31.JPG': ['230', '232','256', '251', '241'], '32.JPG': ['225', '223', '214'], '33.JPG': ['222', '225', '183', '208'],
'34.JPG': ['200', '157', '171', '246', '254', '227', '234'], '35.JPG': ['119', '116', '126','211'], '36.JPG': ['190','133','137','207','210','215','218','198'],
'37.JPG': ['119','125','209','212','217','220','198'], '38.JPG': ['123','218','223','224','230','248'], '39.JPG': ['98','191','190','217','213','218','226','222']
, '40.JPG': ['129','207','206','229','257'], '41.JPG': ['170','166','195','196','220','221','211','183'], '42.JPG': ['208','253','269','273','237','243']
, '43.JPG': ['187','212','195','230','232','192','195','199','194'], '44.JPG': ['159','192','199','200','174','177','186','175']
, '45.JPG': ['195','239','236','210','216','221','215'], '46.JPG': ['176','218','215','186','194'], '47.JPG': ['216','246','244','221','225','224','252']
, '48.JPG': ['206','207','181','186','184','214'], '49.JPG': ['239','242','205','209','254'], '50.JPG': ['199','201','162','220','250','254']
, '51.JPG': ['221','224','244','227','277'], '52.JPG': ['214','227','233'], '53.JPG': ['210','212','175','208','216'], '54.JPG': ['181','176','230','218','219'],
'55.JPG': ['154','151','218','185','190'], '56.JPG': ['124','122','160','170','148','181','192','199'], '57.JPG': ['184','179','211','230','235']}

'''List of taillights for stereo matching in right images'''
stereo_matching_right = {'1.JPG': ['90', '107', '106', '99', '73', '85'], '2.JPG': ['47', '65', '63', '27'], 
'3.JPG': ['72', '88', '82', '81', '78'], '4.JPG': ['66', '72', '42', '58', '57', '50'], 
'5.JPG': ['116', '138', '140', '145', '134', '133', '103'], '6.JPG': ['93', '118', '120', '141', '114', '108', '87'], 
'7.JPG': ['154', '128', '165', '166', '192', '164'], '8.JPG': ['54', '44', '68', '65', '92', '61'], 
'9.JPG': ['131', '144', '145', '161'], '10.JPG': ['93', '101', '103', '100', '118', '119', '136'], 
'11.JPG': ['66', '71', '73', '101', '100'], '12.JPG': ['100', '105', '104', '115', '130'],
'13.JPG': ['50', '53', '69', '70', '80', '81', '85'], '14.JPG': ['31', '35', '44', '47', '49', '54', '61'], 
'15.JPG': ['62', '64', '66', '73', '75', '80', '86', '93'], '16.JPG': ['55', '57', '60', '70', '72', '78', '88'], 
'17.JPG': ['64', '66', '67', '79', '75', '77', '81', '90'], '18.JPG': ['34', '35', '36', '48', '45', '47', '49', '61'], 
'19.JPG': ['35', '54', '55', '59', '69'], '20.JPG': ['34', '36', '38', '40', '49', '53', '62'], 
'21.JPG': ['39', '40', '41', '42', '50', '54', '56', '65'], '22.JPG': ['35', '38', '37', '39', '45', '44', '47', '51', '58'], 
'23.JPG': ['41', '48', '44', '46', '54', '57', '60', '67'], '24.JPG': ['44', '45', '46', '56', '59', '67'], 
'25.JPG': ['55', '57', '58', '60', '70', '71', '74', '82'], '26.JPG': ['36', '38', '40', '42', '54', '55', '56', '59', '69'], 
'27.JPG': ['53', '55', '57', '66', '63', '65', '68', '75'], '29.JPG': ['118', '117'], '30.JPG': ['139', '141'],
'31.JPG': ['105', '107','133', '120', '112'], '32.JPG': ['121', '116', '106'], '33.JPG': ['138', '139', '71', '74']
, '34.JPG': ['92', '47', '55', '110', '111', '97', '102'], '35.JPG': ['96', '51', '52','104'], '36.JPG': ['105','68','66','121','122','124','126','114'],
'37.JPG': ['62','61','115','120','116','118','112'], '38.JPG': ['85','143','140','138','142','150'], '39.JPG': ['116','182','183','200','190','192','194','193']
, '40.JPG': ['73','121','122','131','148'], '41.JPG': ['91','92','103','102','128','131','107','85'], 
'42.JPG': ['66','82','91','94','73','76'], '43.JPG': ['73','92','106','104','80','84','85','81'], '44.JPG': ['77','95','100','98','81','88','91','82']
, '45.JPG': ['107','131','128','111','118','121','116'], '46.JPG': ['173','195','193','169','174'], '47.JPG': ['173','202','201','167','175','168','205']
, '48.JPG': ['153','154','122','131','123'], '49.JPG': ['132','133','104','105','161'], '50.JPG': ['116','117','100','152','166','170']
, '51.JPG': ['103','104','130','102','137'], '52.JPG': ['105','110','111'], '53.JPG': ['100','98','71','93','94'], '54.JPG': ['122','106','160','137','141'],
'55.JPG': ['97','94','137','117','118'], '56.JPG': ['65','61','80','84','73','96','102','104'], '57.JPG': ['48','45','57','61','60']}

'''List of taillights for taillight pairing in left images'''
taillight_pairing_left = {'1.JPG': ['262', '263', '214', '215'], '2.JPG': ['219', '220'], '3.JPG': ['224', '225'], 
'4.JPG': ['140', '142', '131', '132'], '5.JPG': ['213', '219', '199', '200'], '6.JPG': ['220', '228', '211', '213'], 
'7.JPG': ['192', '204', '225', '227', '186', '187'], '8.JPG': ['182', '192', '211', '213', '232', '250', '174', '175'], 
'9.JPG': ['234', '241', '184', '197', '262', '276', '138', '153'], '10.JPG': ['230', '233', '191', '194', '155', '160', '129', '144'], 
'11.JPG': ['214', '218', '235', '237', '164', '181'], '12.JPG': ['131', '141', '164', '166', '177', '181'],
'13.JPG': ['218', '222', '158', '166', '203', '208'], '14.JPG': ['151', '176', '214', '218', '226', '230', '234', '253'], 
'15.JPG': ['178', '190', '233', '237', '241', '255'], '16.JPG': ['152', '167', '225', '232', '238', '255'], 
'17.JPG': ['169', '183', '226', '229', '235', '250'], '18.JPG': ['166', '180', '222', '225', '229', '238'], 
'19.JPG': ['130', '140', '145', '152', '229', '221', '225', '240'], '20.JPG': ['184', '203', '208', '214', '252', '256', '260', '274'], 
'21.JPG': ['160', '182', '180', '187', '237', '241'], '22.JPG': ['153', '181', '184', '190', '240', '243', '246', '258'], 
'23.JPG': ['162', '188', '181', '192', '232', '237', '241', '252'], '24.JPG': ['140', '154', '159', '165', '205', '211', '216', '228'], 
'25.JPG': ['135', '151', '158', '164', '201', '205', '212', '220'], '26.JPG': ['158', '168', '177', '182', '216', '219', '223', '235'], 
'27.JPG': ['149', '160', '167', '176', '228', '232', '237', '246'], '29.JPG': ['213', '215'], '31.JPG': ['254', '256'],
'33.JPG': ['222', '225', '224', '230', '183', '208'], '34.JPG': ['200', '202', '157', '171',' 246', '254', '227', '234'],
'35.JPG': ['116', '116', '211','214'], '36.JPG': ['133','137','207','218'], '37.JPG': ['119','125','209','220'], '38.JPG': ['123','124','223','224','230','248']
, '39.JPG': ['98','100','191','198','218','226'], '40.JPG': ['207','210'], '41.JPG': ['195','196','228','232','220','221'], '41.JPG': ['170','166','195','196','220','221','211']
, '42.JPG': ['208','209','233','240','269','273'], '43.JPG': ['230','232','192','194'], '44.JPG': ['199','200','174','175'], '45.JPG': ['236','239','210','215']
, '46.JPG': ['215','218','186','189'], '47.JPG': ['244','246'], '48.JPG': ['206','207'], '49.JPG': ['205','209','239','242'], '50.JPG': ['199','201','162','167']
, '51.JPG': ['221','224','177','178','274','277','167','168','193','195'], '52.JPG': ['131','132','146','150','191','196','227','233'],
'53.JPG': ['208','216','175','178'], '54.JPG': ['181','185','176','179'], '55.JPG': ['151','152','185','190'], '56.JPG': ['160','170','151','154']
, '57.JPG': ['211','218','230','235']}

'''List of taillights for taillight pairing in right images'''
taillight_pairing_right = {'1.JPG': [], '2.JPG': [], '3.JPG': [], '4.JPG': ['66', '72'], '5.JPG': ['138', '140'], '6.JPG': ['118', '120'], 
'7.JPG': ['165', '166'], '8.JPG': ['68', '65'], '9.JPG': ['144', '145'], '10.JPG': ['100', '103', '118', '119'],
'11.JPG': ['72', '73', '100', '101'], '12.JPG': ['130', '131'], '13.JPG': ['69', '70', '80', '81', '82', '85'], 
'14.JPG': ['47', '49', '54', '61'], '15.JPG': ['75', '80', '86', '93'], '16.JPG': ['78', '88'], '17.JPG': ['75', '77', '81', '90'],
'18.JPG': ['45', '47', '49', '61'], '19.JPG': ['54', '55', '59', '69'], '20.JPG': ['48', '49', '53', '62'], '21.JPG': ['50', '54', '56', '65'],
'22.JPG': ['44', '47', '51', '58'], '23.JPG': ['54', '57', '60', '67'], '24.JPG': ['56', '59'], '25.JPG': ['70', '71', '74', '82'],
'26.JPG': ['55', '56', '59', '69'], '27.JPG': ['63', '65', '68', '75'], '31.JPG': ['119', '120'], '32.JPG': ['121', '125', '114', '116','104', '106']
, '33.JPG': ['138', '139', '71', '74'], '34.JPG': ['47', '55', '110', '111', '97', '102'], '35.JPG': ['51', '52'], '36.JPG': ['66','68','121','126'],
'37.JPG': ['62','61','112','118'], '38.JPG': ['143','142','150','152'], '39.JPG': ['192','194'], '40.JPG': ['130', '131'], '41.JPG': ['103','102','128','131']
, '42.JPG': ['91','94'], '43.JPG': ['104','106','80','81'], '44.JPG': ['98','100','81','82'], '45.JPG': ['128','131','111','116'], '46.JPG': ['193','195']
, '47.JPG': ['202','201'], '48.JPG': ['153','154'], '49.JPG': ['132','133'], '50.JPG': ['116','117'], '51.JPG': ['129','130','103','104'], '52.JPG': ['110','111'],
'53.JPG': ['93','94'], '54.JPG': [], '55.JPG': ['137','142','117','118'], '56.JPG': ['80','84','96','99'], '57.JPG': ['60','61']}

feature_map = {'ImageID': 1, 'ID': 2, 'DB': 3, 'W': 4, 'H': 5, 'V': 6, 'DN': 7, 'DNN': 8, 'S': 9}#, 'HOG': 9, 'SIFT': 10, ''}#, 'DB': 1, 'SR': 2, 'VR': 3, 'DNR': 4]

training_key = {'Image ID': 1, 'IDL': 2, 'IDR': 3, 'DBL': 4, 'WL': 5, 'HL': 6, 'VL': 7, 'DNL': 8, 'DBR': 9,	'WR': 10, 'HR': 11, 'VR': 12, 'DNR': 13, 'NCC': 14, 'M': 15, 'SIFT': 16, 'SURF': 17, 'ORB': 18, 'DNNL': 19, 'DNNR': 20, 'SL': 21, 'SR': 22, 'Vertical ratio': 23, 'Size ratio': 24}

pairing_key = {
'Image ID': 1, 
'IDL': 2, 
'IDR': 3, 
'WL': 4, 
'HL': 5, 
'VL': 6, 
'WR': 7, 
'HR': 8, 
'VR': 9, 
'NCC': 10, 
'SIFT': 11, 
'SURF': 12, 
'ORB': 13, 
'SL': 14, 
'SR': 15, 
'Inter-distance': 16, 
'Vertical ratio': 17, 
'Size ratio': 18, 
'M': 19}

class FD(Enum):
    SIFT = 1
    SURF = 2
    ORB = 3
    DENSE_SIFT = 4
    DENSE_SURF = 5
    DENSE_ORB = 6

class STEREO(Enum):
    LEFT = 0
    RIGHT = 1

class DIM(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    BOTH = 2

min_H_1 = 342 / 2
max_H_1 = 180
min_H_2 = 0
max_H_2 = 9 / 2
max_S = 255
min_S = 0.4645 * max_S
max_V = 255
min_V = 0.2 * max_S
idx = 2
isLeft = False

POS_INF = 999999
NEG_INF = -1
NUM_OF_BIN = 16

DESCRIPTOR_DSIFT = 0
DESCRIPTOR_HOG = 1

# SIZE_THRES = 5
# HEIGHT_THRES = float(2/5)#float(1/2)#standard dataset: 3/4
# BRAKE_THRES = 5

SIZE_THRES = 10#default: 5
HEIGHT_THRES = float(3/5)#float(1/2)#standard dataset: 3/4
BRAKE_THRES = 5