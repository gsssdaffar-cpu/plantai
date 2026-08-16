from database import get_connection

conn = get_connection()
cursor = conn.cursor()

diseases = [

("Pepper__bell___Bacterial_spot","Bell Pepper","Bacterial Spot",
"Bacteria (Xanthomonas spp.)",
"Small water-soaked spots on leaves and fruits",
"Remove infected leaves; Apply copper-based bactericide; Avoid overhead watering",
"Use disease-free seeds; Rotate crops; Keep foliage dry"),

("Pepper__bell___healthy","Bell Pepper","Healthy",
"No disease detected",
"Healthy green leaves",
"No treatment required",
"Continue good agricultural practices"),

("Potato___Early_blight","Potato","Early Blight",
"Alternaria solani fungus",
"Brown concentric rings on older leaves",
"Apply fungicide; Remove infected leaves",
"Crop rotation; Use certified seed potatoes"),

("Potato___Late_blight","Potato","Late Blight",
"Phytophthora infestans",
"Dark water-soaked lesions on leaves",
"Apply fungicide immediately; Remove infected plants",
"Avoid prolonged leaf wetness; Improve air circulation"),

("Potato___healthy","Potato","Healthy",
"No disease detected",
"Healthy green foliage",
"No treatment required",
"Maintain proper irrigation and nutrition"),

("Tomato_Bacterial_spot","Tomato","Bacterial Spot",
"Bacteria (Xanthomonas spp.)",
"Small black leaf spots with yellow halos",
"Copper bactericide; Remove infected leaves",
"Use disease-free seeds; Rotate crops"),

("Tomato_Early_blight","Tomato","Early Blight",
"Alternaria solani fungus",
"Brown target-like spots on lower leaves",
"Apply fungicide; Remove infected leaves",
"Crop rotation; Keep leaves dry"),

("Tomato_Late_blight","Tomato","Late Blight",
"Phytophthora infestans",
"Dark lesions and white fungal growth",
"Apply fungicide; Destroy infected plants",
"Avoid overhead irrigation"),

("Tomato_Leaf_Mold","Tomato","Leaf Mold",
"Passalora fulva fungus",
"Yellow spots with olive mold underneath leaves",
"Apply fungicide; Increase ventilation",
"Reduce humidity inside greenhouse"),

("Tomato_Septoria_leaf_spot","Tomato","Septoria Leaf Spot",
"Septoria lycopersici fungus",
"Numerous small circular spots",
"Remove infected leaves; Apply fungicide",
"Crop rotation"),

("Tomato_Spider_mites_Two_spotted_spider_mite","Tomato",
"Spider Mite",
"Two-spotted spider mite",
"Yellow stippling and webbing",
"Apply miticide; Spray water under leaves",
"Monitor plants regularly"),

("Tomato__Target_Spot","Tomato","Target Spot",
"Corynespora cassiicola fungus",
"Brown circular spots with concentric rings",
"Apply fungicide; Remove infected leaves",
"Improve air circulation"),

("Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato",
"Yellow Leaf Curl Virus",
"Tomato Yellow Leaf Curl Virus",
"Yellow curled leaves and stunted growth",
"Remove infected plants",
"Control whiteflies"),

("Tomato__Tomato_mosaic_virus","Tomato",
"Tomato Mosaic Virus",
"Tomato Mosaic Virus",
"Light and dark green mosaic pattern",
"Remove infected plants",
"Disinfect tools; Use resistant varieties"),

("Tomato_healthy","Tomato","Healthy",
"No disease detected",
"Healthy green leaves",
"No treatment required",
"Maintain healthy farming practices")

]

cursor.executemany("""

INSERT OR REPLACE INTO diseases
(class_name,plant,disease,cause,symptoms,treatment,prevention)
VALUES(?,?,?,?,?,?,?)
""", diseases)

conn.commit()
conn.close()

print("✅ 15 diseases inserted successfully.")