import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_IMGS_DIR = os.path.join(ROOT_DIR, "test_data")
TEST_IMG_1 = os.path.join(TEST_IMGS_DIR, "img1.jpg")
TEST_IMG_2 = os.path.join(TEST_IMGS_DIR, "img1.JPG")
