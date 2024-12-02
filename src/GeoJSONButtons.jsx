import React, { useState } from "react";
import { Download, Copy, Check } from "lucide-react";

const GeoJSONButtons = ({ floor }) => {
    const [copied, setCopied] = useState(false);

    const handleDownload = () => {
        if (floor?.geojson) {
            const blob = new Blob([JSON.stringify(floor.geojson, null, 2)], {
                type: "application/json",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `floor_${floor._id}_geojson.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };

    const handleCopy = async () => {
        if (floor?.geojson) {
            try {
                await navigator.clipboard.writeText(
                    JSON.stringify(floor.geojson, null, 2)
                );
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
            } catch (err) {
                console.error("Failed to copy: ", err);
            }
        }
    };

    return (
        <div className="flex gap-10 pt-4">
            <button
                onClick={handleDownload}
                className="w-full py-3 border border-white rounded-full hover:bg-blue-600 duration-200 flex items-center justify-center gap-2"
            >
                <Download className="w-4 h-4" />
                Download GeoJSON
            </button>

            <button
                onClick={handleCopy}
                className="w-full py-3 border border-white rounded-full hover:bg-blue-600 duration-200 flex items-center justify-center gap-2"
            >
                {copied ? (
                    <>
                        <Check className="w-4 h-4" />
                        Copied!
                    </>
                ) : (
                    <>
                        <Copy className="w-4 h-4" />
                        Copy GeoJSON
                    </>
                )}
            </button>
        </div>
    );
};

export default GeoJSONButtons;
