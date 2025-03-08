import { useState, useEffect } from "react";
import { db } from "../Firebase/Config";
import { collection, getDocs } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

export default function ItemList() {
  const [items, setItems] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchItems = async () => {
      const querySnapshot = await getDocs(collection(db, "items"));
      const itemList = querySnapshot.docs.map((doc) => ({
        id: doc.id, // Firestore doc ID
        ...doc.data(), // Firestore fields
      }));
      setItems(itemList);
    };

    fetchItems();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 w-full">
      <h1 className="text-3xl font-bold text-blue-400">Available Items</h1>
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-5xl">
        {items.map((item) => (
          <div
            key={item.id}
            className="bg-gray-800 p-4 rounded-lg shadow-md hover:bg-gray-700 transition cursor-pointer"
            onClick={() => navigate(`/bill/${item.part_id}`)}
          >
            <h2 className="text-lg font-semibold">{item.name}</h2>
            <p className="text-gray-400">Stock: {item.current_stock}</p>
            <p className="text-gray-400">Lead Time: {item.lead_time}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
