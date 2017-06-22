import face_recognition

known_image = face_recognition.load_image_file("./503/park-geun-hye-3.jpg")
unknown_image = face_recognition.load_image_file("./503/unknown_503/23skorea-1-master768.jpg")

park_encoding = face_recognition.face_encodings(known_image)[0]
unknown_park_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([park_encoding], unknown_park_encoding)
