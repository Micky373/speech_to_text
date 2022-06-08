import React from "react";

import { Link, NavLink } from "react-router-dom";
const Sidebar = () => {
  return (
    <div className="h-screen bg-[#333] w-[500px]">
      <div className="text-white font-bold">
        <div className="text-center mx-9 font-mono text-3xl">
          <h1 className="p-5"> AMHARIC SPEECH TO TEXT</h1>
        </div>
        <hr></hr>
        <div className="flex flex-col mt-24 px-3  text-2xl">
          <Link
            className="w-full mb-2 font-extralight border-2 border-gray-500"
            to={"/"}
          >
            Homepage
          </Link>
          <Link
            className="w-full mb-2 font-extralight border-2 border-gray-500"
            to={"/upload"}
          >
            speech-to -text
          </Link>
          <Link
            className="w-full mb-5 font-extralight border-2 border-gray-500"
            to={"/insights"}
          >
            Insight
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
