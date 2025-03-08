import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { db } from "../Firebase/Config";
import {
  collection,
  addDoc,
  query,
  where,
  getDocs,
  updateDoc,
} from "firebase/firestore";

export default function BillItem() {
  const { part_id } = useParams();
  const [formData, setFormData] = useState({
    part_id: "",
    aircraft_type: "",
    quantity: "",
  });

  useEffect(() => {
    const fetchItemDetails = async () => {
      const itemQuery = query(
        collection(db, "items"),
        where("part_id", "==", part_id)
      );
      const querySnapshot = await getDocs(itemQuery);

      if (!querySnapshot.empty) {
        const itemData = querySnapshot.docs[0].data();
        setFormData((prev) => ({
          ...prev,
          part_id: itemData.part_id,
        }));
      }
    };

    if (part_id) fetchItemDetails();
  }, [part_id]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const date = new Date().toISOString().split("T")[0]; // Current date in YYYY-MM-DD
    const { aircraft_type, quantity } = formData;
    const qty = parseInt(quantity, 10);

    try {
      await addDoc(collection(db, "bills"), {
        ds: date,
        y: qty,
        part_id,
        aircraft_type,
      });

      const itemQuery = query(
        collection(db, "items"),
        where("part_id", "==", part_id)
      );
      const querySnapshot = await getDocs(itemQuery);

      if (querySnapshot.empty) {
        alert("Item not found in database!");
        return;
      }

      const itemDoc = querySnapshot.docs[0];
      const itemRef = itemDoc.ref;
      const currentStock = parseInt(itemDoc.data().current_stock, 10);

      if (isNaN(currentStock)) {
        alert("Invalid stock quantity in database!");
        return;
      }

      const newStock = Math.max(currentStock - qty, 0);
      await updateDoc(itemRef, { current_stock: newStock });

      alert("Bill added and stock updated successfully!");
      setFormData({ part_id: "", aircraft_type: "", quantity: "" });
    } catch (error) {
      console.error("Error updating stock: ", error);
      alert("Failed to add bill or update stock.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 pt-16 w-full">
      <h1 className="text-3xl font-bold text-blue-400">Bill Item</h1>
      <form
        onSubmit={handleSubmit}
        className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-3xl mt-6"
      >
        {Object.keys(formData).map((key) => (
          <div key={key} className="mb-4">
            <label className="block text-gray-300 capitalize">
              {key.replace("_", " ")}
            </label>
            <input
              type={key === "quantity" ? "number" : "text"}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="w-full p-2 mt-1 bg-gray-700 text-white border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
              required
              disabled={key === "part_id"} // Make part_id readonly
            />
          </div>
        ))}
        <button
          type="submit"
          className="w-full bg-blue-500 p-2 rounded mt-4 hover:bg-blue-600 transition-all"
        >
          Add Bill
        </button>
      </form>
    </div>
  );
}
