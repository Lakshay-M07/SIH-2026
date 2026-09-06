/**
 * Firebase Admin Initialization Stub / Safe Configuration Loader
 * Day 1 Isolation Principle:
 * Allows the server to run locally without crashing when Firebase credentials
 * are not yet provisioned.
 */

let firebaseApp = null;
let db = null;

export const initFirebase = () => {
  const projectId = process.env.FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const privateKey = process.env.FIREBASE_PRIVATE_KEY;

  if (!projectId || !clientEmail || !privateKey) {
    console.warn(
      "[Firebase] Running in decoupled/unconfigured mode. Placeholders remain in .env.example."
    );
    return { app: null, db: null, isConfigured: false };
  }

  // When credentials are set in future days, firebase-admin will initialize here
  return { app: firebaseApp, db, isConfigured: false };
};

export const getFirestore = () => db;
