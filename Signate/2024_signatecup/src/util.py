import os

def create_directory_if_not_exists(directory_path):
    """
    指定されたパスにディレクトリが存在しない場合、新しいディレクトリを作成する関数。
    
    Parameters:
    ----------
    directory_path : str
        作成するディレクトリのパス。
    """
    # ディレクトリが存在しない場合、新規作成
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        # 既に存在する場合、その旨を通知
        print(f"Directory '{directory_path}' already exists.")
