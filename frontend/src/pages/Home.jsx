export default function Home() {
  const cards = [
    { title: "Add Item", route: "/add-item" },
    { title: "Bill Item", route: "/bill-item" },
    { title: "Predict Usage", route: "/predict-usage" },
    { title: "Add Usage", route: "/add-usage" },
  ];

  return (
    <div
      className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-8 pt-20 bg-cover bg-center bg-no-repeat"
      style={{
        backgroundImage:
          "url('https://bsmedia.business-standard.com/_media/bs/img/article/2024-08/07/full/1722995892-9811.jpg?im=FeatureCrop,size=(826,465)')",
        backgroundBlendMode: "overlay",
        backgroundColor: "rgba(17, 24, 39, 0.85)", // Semi-transparent dark overlay
      }}
    >
      <div className="max-w-5xl w-full flex flex-col items-center">
        <h1 className="text-4xl font-bold text-blue-400">Flight Maintenance</h1>
        <p className="text-lg text-gray-300 mt-2 text-center">
          Efficiently manage and maintain flight components
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6 mt-8 w-full justify-items-center">
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
    </div>
  );
}
