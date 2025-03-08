export default function Home() {
  const cards = [
    { title: "Add Item", route: "/add-item" },
    { title: "Update Item", route: "/update-item" },
    { title: "Predict Usage", route: "/predict-usage" },
    { title: "Add Usage", route: "/add-usage" },
  ];

  return (
    <div className="min-h-screen bg-transparent text-white flex flex-col items-center p-8 pt-20">
      <h1 className="text-4xl font-bold text-blue-400">Flight Maintenance</h1>
      <p className="text-lg text-gray-300 mt-2 text-center">
        Efficiently manage and maintain flight components
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6 mt-8">
        {cards.map((card, index) => (
          <a
            key={index}
            href={card.route}
            className="bg-gray-800 shadow-lg rounded-xl p-6 text-center hover:bg-blue-500 transition-all w-64 sm:w-80 lg:w-96"
          >
            <h2 className="text-xl font-semibold text-white">{card.title}</h2>
          </a>
        ))}
      </div>
    </div>
  );
}
