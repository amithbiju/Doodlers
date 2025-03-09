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
      return "text-red-600 font-bold";
    } else if (daysLeft < 15) {
      return "text-yellow-600 font-bold";
    } else {
      return "text-green-600";
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
    return <div className="text-center p-4">Loading orders data...</div>;
  }

  if (error) {
    return <div className="text-center p-4 text-red-600">{error}</div>;
  }

  return (
    <div className="w-full min-h-screen p-4 pt-32 bg-gray-900">
      <h1 className="text-2xl font-bold mb-4 text-white">
        Parts Inventory & Reorder List
      </h1>

      {orders.length === 0 ? (
        <div className="text-center p-4 text-white">No orders found</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-gray-900 shadow-md rounded-2xl border border-white">
            <thead className="bg-gray-900 border-b border-white">
              <tr>
                <th className="py-3 px-4 text-left text-white border-r  border-white">
                  Part ID
                </th>
                <th className="py-3 px-4 text-left text-white border-r border-white">
                  Current Stock
                </th>
                <th className="py-3 px-4 text-left text-white border-r border-white">
                  Min Stock
                </th>
                <th className="py-3 px-4 text-left text-white border-r border-white">
                  Days Left
                </th>
                <th className="py-3 px-4 text-left text-white border-r border-white">
                  Lead Time
                </th>
                <th className="py-3 px-4 text-left text-white border-r border-white">
                  Reorder Date
                </th>
                <th className="py-3 px-4 text-left text-white border-l border-white">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {orders.map((order) => (
                <tr key={order.part_id} className="border-t border-white">
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {order.part_id}
                  </td>
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {order.current_stock}
                  </td>
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {order.min_stock}
                  </td>
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {order.days_left}
                  </td>
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {order.lead_time} days
                  </td>
                  <td className="py-3 px-4 text-gray-300 border-r border-white">
                    {formatDate(order.reorder_date)}
                  </td>
                  <td
                    className={`py-3 px-4 border-white ${getStatusColor(
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
