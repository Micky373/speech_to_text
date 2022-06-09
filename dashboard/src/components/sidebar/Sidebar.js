import React from "react";

import { Link, NavLink } from "react-router-dom";
const Sidebar = () => {
  return (
    <div className="h-screen bg-gradient-to-r from-slate-900 to-slate-700 w-[500px]">
      <div className="text-white font-bold">
        <div className="text-center mx-9 font-mono text-3xl">
          <h1 className="p-5"> AMHARIC SPEECH TO TEXT</h1>
        </div>
        <hr></hr>
        <div className="flex flex-col mt-24 px-3  text-2xl">
          <hr></hr>
          <Link className="w-full mb-2 font-extralight  " to={"/"}>
            Homepage
          </Link>
          <hr></hr>
          <Link className="w-full mb-2 font-extralight  " to={"/upload"}>
            speech-to -text
          </Link>
          <hr></hr>
          <Link className="w-full font-extralight  " to={"/insights"}>
            Insight
          </Link>
          <hr></hr>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
