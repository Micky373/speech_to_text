import React, { useEffect } from "react";

function Upload() {
  useEffect(() => {
    fetch("/output").then((response) =>
      response.json().then((data) => {
        console.log(data);
      })
    );
  }, []);

  return (
    <div className="w-full h-full pt-10">
      <h1 className="text-center">Upload your sound file here</h1>
      <div className="flex flex-col h-full  items-center pt-10">
        <div className="border-5 w-[700px] mt-10 p-2">
          <h1 className="text-center">Upload file</h1>
        </div>
        <div className="border-5 w-[700px] h-[700px] mt-10 pt-10"></div>
      </div>
    </div>
  );
}

export default Upload;
