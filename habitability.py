import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('hwc.csv')
df.dropna(subset=['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ESI'], inplace=True)

features = ['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ESI', 
            'P_SEMI_MAJOR_AXIS', 'P_MASS', 'P_RADIUS']

X = df[features]
y = df['P_HABITABLE']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
