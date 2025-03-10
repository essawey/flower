def download():
    import os
    if not os.path.exists(os.path.join(os.getcwd(), 'PanNuke', 'data', 'PanNuke.zip')):

        target_path = os.path.join(os.getcwd(),"PanNuke", "data")

        zip_path = os.path.join(target_path, "PanNuke.zip")
        os.makedirs(target_path, exist_ok=True)

        import gdown
        id = "1-lyR2TY30Y-k_Tz1gs0RK8FzojMYedsN"
        gdown.download(id=id, output=zip_path)


        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)


        from PanNuke import transforms
        transforms.create_patches(target_path, "Patched")


        import shutil
        # Define the current working directory
        current_dir = os.getcwd()

        # Move "Patched/PanNuke" to "Patched/Patched"
        src = os.path.join(current_dir, "Patched", "PanNuke")
        dst = os.path.join(current_dir, "Patched", "Patched")
        shutil.move(src, dst)

        # Move "Patched/Patched" to "PanNuke/data"
        src = os.path.join(current_dir, "Patched", "Patched")
        dst = os.path.join(current_dir, "PanNuke", "data")
        shutil.move(src, dst)

        # Remove the "Patched" directory
        patched_dir = os.path.join(current_dir, "Patched")
        os.rmdir(patched_dir)

        # Move "PanNuke" to "Original" in the target path
        src = os.path.join(target_path, "PanNuke")
        dst = os.path.join(target_path, "Original")
        shutil.move(src, dst)


download()
