
import os
import csv
import shutil
import cv2

def main(folderPath, prefix = "CADICA", outputFolder = "dataset"):
    
    #Validate that the orginal name CSV files exists
    if(not os.path.exists("originalNames.csv")): 
        with open("originalNames.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["New Name", "Old Name", "Path", "Dataset"])
            
    if(not os.path.exists("imageStates.csv")): 
        with open("imageStates.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "State"])
    
    
    count = 0
    
    orginalNames = []
    imageStates = []
    
    
    for root, dirs, files in os.walk(folderPath):
        modifedRoot = root.replace(folderPath, "")
        sortedFiles = sorted(files)
        for file in sortedFiles:
            if(file.endswith(".png") or file.endswith(".jpg")):
                hexString = str(count).zfill(6)
                newFileName = f"{prefix}_{hexString}.jpg"
                orginalNames.append([newFileName, file, modifedRoot, prefix])
                imageStates.append([newFileName, "Labeled"])
                img = cv2.imread(os.path.join(root, file))
                cv2.imwrite(os.path.join(outputFolder, newFileName), img)
                count += 1
    
    
    with open("originalNames.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerows(orginalNames)
    
    with open("imageStates.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerows(imageStates)

if(__name__ == "__main__"):
    folderPath = "CSAngioImages"
    main(folderPath, prefix="CSAngioImages", outputFolder="dataset")