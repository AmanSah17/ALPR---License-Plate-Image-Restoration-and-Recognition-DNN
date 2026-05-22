from utils import utils
import argparse

def main():
    parser = argparse.ArgumentParser(description='Folder Location')
    parser.add_argument('--LR', type= str, default="lr_plate_crop")
    parser.add_argument('--HR', type= str, default="hr_plate_crop")
    args = parser.parse_args()

    base_folder = "data"
    
    # Copy folders (lr_plate_crop) -> Homography Transformation -> Select ROI
    # LR Path e.g: plate_crop (1~31) // HR Path e.g: HR_plate_crop (Requirement: Fixed file name = 'HR.png')
    
    utils.initial_settings(base_folder, args.LR)
    utils.Homography_warp(base_folder, args.LR, args.HR)


if __name__ == "__main__":
    main()
    
