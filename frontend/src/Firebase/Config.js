// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyChUwjxcu_LbPRdux0ISFRXJyZUYHXM7gg",
  authDomain: "flight-maintenance.firebaseapp.com",
  projectId: "flight-maintenance",
  storageBucket: "flight-maintenance.firebasestorage.app",
  messagingSenderId: "783251558168",
  appId: "1:783251558168:web:a79d431c8c0f674e7e586d",
  measurementId: "G-3FZZK2KTNR",
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
const analytics = getAnalytics(app);
