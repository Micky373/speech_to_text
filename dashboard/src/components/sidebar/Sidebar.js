import React from "react";
import {
  CDBSidebar,
  CDBSidebarContent,
  CDBSidebarFooter,
  CDBSidebarHeader,
  CDBSidebarMenu,
  CDBSidebarMenuItem,
} from "cdbreact";
import { Link, NavLink } from "react-router-dom";
const Sidebar = () => {
  return (
    <div className="h-screen w-[300px]">
      <div className="bg-[#333] h-full text-white">
        <div className="text-center p-9 font-serif text-2xl">
          <h1> AMHARIC SPEECH TO TEXT</h1>
        </div>
        <hr></hr>
        <div className="flex flex-col w-full h-full pt-28 pl-8 text-xl">
          <Link className="pb-10" to={"/"}>
            Homepage
          </Link>
          <Link to={"/"}>Insight</Link>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
