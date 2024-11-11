import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score



df = pd.read_csv('hwc.csv')
df.dropna(subset=['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ESI'], inplace=True)

def is_habitable(exoplanet):
    if 0 <= exoplanet['P_TEMP_SURF'] <= 100 and exoplanet['P_TEMP_EQUIL'] >= exoplanet['S_HZ_OPT_MIN'] \
    and exoplanet['P_TEMP_EQUIL'] <= exoplanet['S_HZ_OPT_MAX']:
        return 1  # Habitable
    return 0  # Non-habitable

df['Habitability'] = df.apply(is_habitable, axis=1)
features = ['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ESI', 
    'P_SEMI_MAJOR_AXIS', 'P_MASS', 'P_RADIUS']

X = df[features]
y = df['Habitability']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
