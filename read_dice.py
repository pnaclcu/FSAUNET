import os


def compute_dice(base_dir):
    all_time = os.listdir(base_dir)
    for time in all_time:
        time_path = os.path.join(base_dir, time, 'FSAUNet')
        all_ckpts = os.listdir(time_path)
        for ckpt in all_ckpts:
            cktp_path = os.path.join(time_path, ckpt)
            mask_dir_path = os.path.join(cktp_path, 'mask')
            all_mask = os.listdir(mask_dir_path)
            img_num = len(all_mask)
            score_mask = 0
            score_con = 0
            es_dice = 0
            ed_dice = 0
            if 'CAMUS' in base_dir:
                for mask in all_mask:
                    score = float(mask.split('_')[-1].split('.png')[0])
                    score_mask += score
                    if 'ES' in mask:
                        es_dice += score
                    else:
                        ed_dice += score
                print('ckpt path:{},mask_dice:{},con_dice:{},es_dice:{},ed_dice:{},'.format(
                    cktp_path, score_mask / img_num, score_con / img_num, es_dice * 2 / img_num, ed_dice * 2 / img_num,
                    ))
if __name__ == '__main__':

    compute_dice('./CAMUS_perdiction')