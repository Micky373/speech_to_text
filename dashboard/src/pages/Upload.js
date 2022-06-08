import React, { useEffect, useRef, useState } from "react";
import UploadAudio from "../components/UploadAudio";
import axios from "axios";

var URLst = "";

function Upload() {
  const inputFile = useRef(null);

  const [text, setText] = useState("");
  const [selectedFile, setFile] = useState("");
  const onButtonClick = () => {
    // `current` points to the mounted file input element
    var c = inputFile.current.click();
    console.log(c);
    this.postData();
    // this.getData();
  };
  const fileChangedHandler = (event) => {
    console.log(event.target.files);
    setFile(event.target.files[0]);
  };

  function postData() {
    var formData = new FormData();
    formData.append("file", selectedFile);
    axios({
      method: "Post",
      url: "https://afri-speech-to-text.herokuapp.com/get_file",
      data: formData,
      headers: { "Content-Type": "multi-part/form-data" },
    })
      .then((res) => {
        console.log(res.data);
        setText(res.data.success);
      })
      .catch(function (error) {
        console.log(error);
      });
  }

  return (
    <div className=" h-full w-full pt-10 flex items-center justify-center">
      <div className="">
        <h1 className="text-center">Upload your sound file here</h1>
        <div className="flex flex-col h-full  items-center pt-10">
          <div className="border-5 w-[700px] flex justify-center mt-10 p-2"></div>

          <form>
            <input
              type="file"
              name="file"
              onChange={(e) => {
                fileChangedHandler(e);
              }}
            />
          </form>
          <button
            className="text-center"
            onClick={() => {
              postData();
            }}
          >
            Upload file 📁
          </button>

          <div className="border-5 w-[700px] h-[700px] mt-10 pt-10">
            <p className="text-gray-500 mx-1">Pridicted Response is: </p>
            <p>{text}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Upload;
