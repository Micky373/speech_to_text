import React from "react";

import { Link, NavLink } from "react-router-dom";
const Sidebar = () => {
  return (
    <div className="h-screen w-[500px]">
      <div className="bg-[#333] h-full text-white font-bold">
        <div className="text-center pt-9 pb-9 font-mono text-3xl">
          <h1 className="p-2"> AMHARIC SPEECH TO TEXT</h1>
        </div>
        <hr></hr>
        <div className="flex flex-col w-full h-full pt-28  text-2xl">
          <Link
            className="w-full mb-2 font-extralight border-2 border-gray-500"
            to={"/upload"}
          >
            Homepage
          </Link>
          <Link
            className="w-full mb-5 font-extralight border-2 border-gray-500"
            to={"/"}
          >
            Insight
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
