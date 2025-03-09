import React, { useState, useEffect } from "react";
import axios from "axios";

const Predict = () => {
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Function to fetch orders data
    const fetchOrders = async () => {
      try {
        setLoading(true);
        const response = await axios.get("http://127.0.0.1:5000/get_orders");
        setOrders(response.data);
        setError(null);
      } catch (err) {
        setError("Failed to fetch orders. Please try again later.");
        console.error("Error fetching orders:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchOrders();
  }, []);

  // Function to determine status color based on days_left and current_stock
  const getStatusColor = (daysLeft, currentStock, minStock) => {
    if (daysLeft < 0 || currentStock <= minStock) {
      return "text-red-400 font-bold";
    } else if (daysLeft < 15) {
      return "text-yellow-400 font-bold";
    } else {
      return "text-green-400";
    }
  };

  // Format reorder date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  if (loading) {
    return (
      <div className="text-center p-4 text-white">Loading orders data...</div>
    );
  }

  if (error) {
    return <div className="text-center p-4 text-red-400">{error}</div>;
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-8 pt-16 w-full">
      <h1 className="text-3xl font-bold text-blue-400 mb-6">
        Parts Inventory & Reorder List
      </h1>

      {orders.length === 0 ? (
        <div className="text-center p-4 text-gray-300">No orders found</div>
      ) : (
        <div className="overflow-x-auto w-full max-w-3xl">
          <table className="min-w-full bg-gray-800 shadow-lg rounded-lg overflow-hidden">
            <thead className="bg-gray-700 text-gray-300">
              <tr>
                <th className="py-3 px-4 text-left">Part ID</th>
                <th className="py-3 px-4 text-left">Current Stock</th>
                <th className="py-3 px-4 text-left">Min Stock</th>
                <th className="py-3 px-4 text-left">Days Left</th>
                <th className="py-3 px-4 text-left">Lead Time</th>
                <th className="py-3 px-4 text-left">Reorder Date</th>
                <th className="py-3 px-4 text-left">Status</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order) => (
                <tr key={order.part_id} className="border-t border-gray-700">
                  <td className="py-3 px-4">{order.part_id}</td>
                  <td className="py-3 px-4">{order.current_stock}</td>
                  <td className="py-3 px-4">{order.min_stock}</td>
                  <td
                    className={`py-3 px-4 ${
                      order.days_left < 0 ? "text-red-400" : ""
                    }`}
                  >
                    {order.days_left}
                  </td>
                  <td className="py-3 px-4">{order.lead_time} days</td>
                  <td className="py-3 px-4">
                    {formatDate(order.reorder_date)}
                  </td>
                  <td
                    className={`py-3 px-4 ${getStatusColor(
                      order.days_left,
                      order.current_stock,
                      order.min_stock
                    )}`}
                  >
                    {order.days_left < 0
                      ? "Overdue"
                      : order.current_stock <= order.min_stock
                      ? "Critical Stock"
                      : order.days_left < 15
                      ? "Order Soon"
                      : "In Stock"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Predict;
