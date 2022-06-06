import React, { useEffect, useState } from "react";
import axios from "axios";

function UploadAudio() {
  const [person, setPerson] = useState();

  useEffect(() => {
    axios.get(`https://jsonplaceholder.typicode.com/users`).then((res) => {
      console.log(res);
    });
  }, []);

  return <div>UploadAudio</div>;
}

export default UploadAudio;
