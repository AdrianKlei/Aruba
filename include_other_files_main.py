from include_file_1 import *

if __name__ == '__main__':
    print("Hello my friend.")
    general_information_output()
    second_function()

    errors = [10, 20, 5, 40, 3, 7]
    print(errors)
    print("chosen order %d" %errors.index(min(errors)))