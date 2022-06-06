import React, { useEffect, useRef, useState } from "react";
import UploadAudio from "../components/UploadAudio";

function Upload() {
  const inputFile = useRef(null);

  const [text, setText] = useState("");

  useEffect(() => {
    fetch("/output").then((response) =>
      response.json().then((data) => {
        setText(data);
      })
    );
  }, []);

  const onButtonClick = () => {
    // `current` points to the mounted file input element
    inputFile.current.click();
  };

  return (
    <div className="w-full h-full pt-10">
      <h1 className="text-center">Upload your sound file here</h1>
      <div className="flex flex-col h-full  items-center pt-10">
        <div className="border-5 w-[700px] flex justify-center mt-10 p-2">
          <button className="text-center" onClick={onButtonClick}>
            Upload file 📁
          </button>
          <input
            type="file"
            id="file"
            ref={inputFile}
            style={{ display: "none" }}
          />
        </div>
        <div className="border-5 w-[700px] h-[700px] mt-10 pt-10">
          <p>{text}</p>
        </div>
        <UploadAudio />
      </div>
    </div>
  );
}

export default Upload;
