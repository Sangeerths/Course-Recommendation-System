import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
df = pd.read_csv('courselist.csv')
df_courses = df[['Title', 'COURSE CATEGORIES']].drop_duplicates()
df_courses['COURSE CATEGORIES'] = df_courses['COURSE CATEGORIES'].fillna('')
df_courses['COURSE CATEGORIES'] = df_courses['COURSE CATEGORIES'].str.lower()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df_courses['COURSE CATEGORIES'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

app = FastAPI()

def get_recommendations(skill, cosine_sim=cosine_sim):
    normalized_skill = skill.lower().replace(" ", "")
    mask = df_courses['Title'].str.lower().str.replace(" ", "").str.contains(normalized_skill)
    matching_courses = df_courses[mask]

    if matching_courses.empty:
        logger.info(f"No courses found for the skill: {skill}")
        return f"No courses found for the skill: {skill}"
    matching_indices = matching_courses.index.tolist()
    sim_scores = []
    for idx in matching_indices:
        if idx < cosine_sim.shape[0]:
            sim_scores.append((idx, cosine_sim[idx].mean()))  

    if not sim_scores:
        logger.info(f"No valid courses found for the skill: {skill}")
        return f"No valid courses found for the skill: {skill}"
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:5]]  # Top 5 courses
    return df_courses['Title'].iloc[top_indices].tolist()

@app.get("/recommend")
def recommend_courses(skill: str):
    try:
        recommendations = get_recommendations(skill)
        if isinstance(recommendations, str):  
            raise HTTPException(status_code=404, detail=recommendations)
        
        return {"Recommended Courses": recommendations}
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# To run the application, use the command:
# uvicorn main:app --reload
