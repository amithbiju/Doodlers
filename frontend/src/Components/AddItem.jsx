import { useState } from "react";
import { db } from "../Firebase/Config";
import { collection, addDoc } from "firebase/firestore";

export default function AddItem() {
  const [formData, setFormData] = useState({
    part_id: "",
    name: "",
    description: "",
    current_stock: "",
    lead_time: "",
    min_stock: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;

    // Convert current_stock and min_stock to numbers
    if (name === "current_stock" || name === "min_stock") {
      setFormData({ ...formData, [name]: value === "" ? "" : Number(value) });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Create a copy of the form data with appropriate numeric conversions
    const submissionData = {
      ...formData,
      current_stock: Number(formData.current_stock),
      min_stock: Number(formData.min_stock),
    };

    try {
      await addDoc(collection(db, "items"), submissionData);
      alert("Item added successfully!");
      setFormData({
        part_id: "",
        name: "",
        description: "",
        current_stock: "",
        lead_time: "",
        min_stock: "",
      });
    } catch (error) {
      console.error("Error adding item: ", error);
      alert("Failed to add item.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 w-full">
      <h1 className="text-3xl font-bold text-blue-400">Add Item</h1>
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
              type={
                key === "current_stock" || key === "min_stock"
                  ? "number"
                  : "text"
              }
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="w-full p-2 mt-1 bg-gray-700 text-white border border-gray-600 rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
              required
            />
          </div>
        ))}
        <button
          type="submit"
          className="w-full bg-blue-500 p-2 rounded mt-4 hover:bg-blue-600 transition-all"
        >
          Add Item
        </button>
      </form>
    </div>
  );
}
