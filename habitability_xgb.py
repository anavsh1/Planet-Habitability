import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv('hwc.csv')
df.dropna(subset=['P_TEMP_SURF', 'P_TEMP_EQUIL','P_ESI'], inplace=True)

features = ['P_TEMP_SURF', 'P_TEMP_EQUIL', 'P_ESI', 
            'P_SEMI_MAJOR_AXIS', 'P_MASS', 'P_RADIUS']
X = df[features]
y = df['P_HABITABLE']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,  
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train_smote, y_train_smote)
y_pred = xgb_model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

solar_system_planets = pd.DataFrame({
    'Planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'],
    'P_TEMP_SURF': [440, 737, 288, 210, 165, 134, 76, 72],  # Surface Temp (K)
    'P_TEMP_EQUIL': [449, 328, 279, 226, 122, 90, 64, 51],  # Equilibrium Temp (K)
    'P_ESI': [0.6, 0.44, 1.00, 0.7, 0.29, 0.25, 0.19, 0.18],  # ESI index
    'P_SEMI_MAJOR_AXIS': [0.39, 0.72, 1.00, 1.52, 5.20, 9.53, 19.18, 30.07],  # AU
    'P_MASS': [0.0553, 0.815, 1.00, 0.107, 317.8, 95.2, 14.5, 17.1],  # Earth mass
    'P_RADIUS': [0.383, 0.949, 1.00, 0.532, 11.21, 9.45, 4.01, 3.88]  # Earth radius
})

X_solar = solar_system_planets.drop(columns=['Planet'])
X_scaled_solar = scaler.transform(X_solar)

habitability_predictions = xgb_model.predict(X_scaled_solar)
solar_system_planets['Habitability_Label'] = habitability_predictions

print(solar_system_planets[['Planet', 'Habitability_Label']])