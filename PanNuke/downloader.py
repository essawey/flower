def download():
    import os
    if not os.path.exists(os.path.join(os.getcwd(), 'PanNuke', 'data', 'PanNuke.zip')):
        import subprocess

        target_path = os.path.join(os.getcwd(),"PanNuke", "data")
        zip_path = os.path.join(target_path, "PanNuke.zip")

        os.makedirs(target_path, exist_ok=True)

        # subprocess.run(["gdown", "-q", "1vVLVUV4hMDovRpItYnogh6cwcfb1xOo5", "-O", zip_path])

        
        subprocess.run(["unzip", "-q", zip_path, "-d", target_path])

        from PanNuke import transforms
        transforms.create_patches(target_path, 192, "Patched")
        
        subprocess.run(["mv", os.path.join(os.getcwd(), "Patched", "PanNuke"),  os.path.join(os.getcwd(), "Patched", "Patched")])
        subprocess.run(["mv", os.path.join(os.getcwd(), "Patched", "Patched"), os.path.join(os.getcwd(), "PanNuke", 'data')])
        subprocess.run(["rmdir", os.path.join(os.getcwd(), "Patched")])
        subprocess.run(["mv", os.path.join(target_path, "PanNuke"), os.path.join(target_path, "Original")])

# print("Downloading PanNuke")
download()
# print("PanNuke downloaded")
