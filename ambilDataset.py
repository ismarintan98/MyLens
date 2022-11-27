import cv2 as cv
import numpy as np
import os


path = "dataset"

cap = cv.VideoCapture(0)

print("------ Program Ambil Dataset ------")
print("tekan 'q' untuk keluar")
print("tekan 's' untuk menyimpan gambar")

idx = 40


while(True):
    
    ret, frame = cap.read()

  
    resized = cv.resize(frame, (384, 288))

    #crop 80%
    height, width, channel = resized.shape
    crop = resized[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]
    

    # cv.imshow('frame', resized)
    cv.imshow('crop', crop)

    #if press 's' save image
    if cv.waitKey(1) & 0xFF == ord('s'):
        while(True):
            print("Masukkan label: ")
            label = input()
            print("Apakah label yang dimasukkan sudah benar? (y/n)")
            confirm = input()
            if confirm == "y":
                cv.imwrite(path + "/" + str(idx)  +  ".png", crop)
                with open(path + "/" + str(idx)  +  ".txt", "w") as f:
                    f.write(label)

                print("Gambar berhasil disimpan")
                idx += 1
                break
            elif confirm == "n":
                print("Mulai ulang")
                break
            else:
                print("Input salah")
                print("Mulai ulang")
                break



    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

    

